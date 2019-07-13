"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created Date: Monday, June 17th 2019, 5:39:09 pm
Copyright: Tommi Hyppänen
"""


import bpy  # noqa:F401
import numpy as np  # noqa:F401
import bmesh  # noqa:F401
from collections import OrderedDict  # noqa:F401
import mathutils as mu  # noqa:F401


# import/reload all source files
if "afm" not in locals():
    from .bpy_amb import utils as au
    from .bpy_amb import fastmesh as afm
    from .bpy_amb import bbmesh as abm
else:
    import importlib

    importlib.reload(au)
    importlib.reload(afm)
    importlib.reload(abm)


try:
    import numba  # noqa:F401
    import scipy  # noqa:F401
except ModuleNotFoundError:
    print("Numba/Scipy not found, trying to install...")
    from subprocess import call

    pp = bpy.app.binary_path_python

    call([pp, "-m", "ensurepip", "--user"])
    call([pp, "-m", "pip", "install", "--user", "numba"])
    call([pp, "-m", "pip", "install", "--user", "scipy"])


def pp_opname(parent, prefix, p):
    return parent + "_" + prefix + "_" + p


class Mesh_Operator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}
    my_props = []
    prefix = ""
    parent_name = ""

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def payload(self, mesh, context):
        pass

    def invoke(self, context, event):
        # self.pr = au.profiling_start()

        # copy property values from panel to operator
        print(self.prefix, self.my_props)
        if self.prefix != "":
            for p in self.my_props:
                opname = pp_opname(self.parent_name, self.prefix, p)
                panel_value = getattr(context.scene, opname)
                setattr(self, p, panel_value)
            print(self.bl_idname)

        return self.execute(context)

    def execute(self, context):
        # apply modifiers for the active object before mesh actions
        for mod in context.active_object.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError as ex:
                print(ex)

        # run mesh operation
        mesh = context.active_object.data
        self.payload(mesh, context)
        # mesh.update(calc_edges=True)

        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column()

        for p in self.my_props:
            row = col.row()
            row.prop(self, p, expand=True)


def mesh_operator_factory(props, prefix, payload, name, parent_name, info):
    temp = type(
        name,
        (Mesh_Operator,),
        {
            "bl_idname": "object." + parent_name + "_" + prefix,
            "bl_label": " ".join(prefix.split("_")).capitalize(),
            "bl_description": info,
            "my_props": props.keys(),
            "prefix": prefix,
            "parent_name": parent_name,
            "payload": payload,
        },
    )
    setattr(temp, "__annotations__", {})
    for k, v in props.items():
        temp.__annotations__[k] = v
    return temp


class Mesh_Master_OP:
    def generate(self):
        pass

    def __init__(self):
        self.props = OrderedDict()
        self.parent_name = "mesh_refine_toolbox"

        self.payload = lambda a, b, c: 0
        self.prefix = ""
        self.name = ""
        self.info = ""
        self.category = ""
        self.fastmesh = False

        self.generate()

        # wrap Bmesh
        if not self.fastmesh:

            def _bm_from_selected(ifunc):
                def _f(this, mesh, context):
                    with abm.Bmesh_from_edit(mesh) as bm:
                        ifunc(this, bm, context)

                return _f

            self.payload = _bm_from_selected(self.payload)

            # wrap mode switching
            def _mode_switch(ifunc):
                def _f(this, mesh, context):
                    mode = context.object.mode
                    bpy.ops.object.mode_set(mode="EDIT")
                    ifunc(this, mesh, context)
                    bpy.ops.object.mode_set(mode=mode)

                return _f

            self.payload = _mode_switch(self.payload)

        self.op = mesh_operator_factory(
            self.props, self.prefix, self.payload, self.name, self.parent_name, self.info
        )
