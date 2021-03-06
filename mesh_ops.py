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

from .bpy_amb import bbmesh as abm  # noqa:F401
from .bpy_amb import master_ops

import importlib

importlib.reload(abm)
importlib.reload(master_ops)


def create(load_these):
    # self,
    # master_name,
    # input_ops,
    # p_panel_draw,
    # p_panel_props,
    # p_idname,
    # p_spacetype,
    # p_regiontype,
    # p_category,
    # additional_classes,
    pbuild = master_ops.PanelBuilder(
        "mesh_toolbox", load_these, {}, {}, "OBUILD", "VIEW_3D", "UI", "Edit", []
    )
    return pbuild.register_params, pbuild.unregister_params


class MeshOperator(master_ops.MacroOperator):
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def payload(self, mesh, context):
        pass

    def execute(self, context):
        # run mesh operation
        mesh = context.active_object.data
        self.payload(mesh, context)
        # mesh.update(calc_edges=True)

        return {"FINISHED"}


class MeshOperatorGenerator(master_ops.OperatorGenerator):
    def __init__(self, master_name):
        self.init_begin(master_name)
        self.fastmesh = False
        self.apply_modifiers = True
        self.generate()
        self.init_end()
        self.name = "OBJECT_OT_" + self.name

        # apply mods
        if self.apply_modifiers:

            def _apply(ifunc):
                def _f(this, mesh, context):
                    # apply modifiers for the active object before mesh actions
                    for mod in context.active_object.modifiers:
                        try:
                            bpy.ops.object.modifier_apply(modifier=mod.name)
                        except RuntimeError as ex:
                            print(ex)
                    ifunc(this, mesh, context)

                return _f

            self.payload = _apply(self.payload)

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

        self.create_op(MeshOperator, "object")
