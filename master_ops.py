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

print("Import: master_ops.py")

import bpy  # noqa:F401
import numpy as np  # noqa:F401
import bmesh  # noqa:F401
from collections import OrderedDict  # noqa:F401
import mathutils as mu  # noqa:F401


class MacroOperator(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}
    my_props = []
    prefix = ""
    parent_name = ""


class OperatorGenerator:
    def generate(self):
        pass

    def init_begin(self, master_name):
        self.props = OrderedDict()
        self.parent_name = master_name

        self.payload = None
        self.prefix = ""
        self.info = ""
        self.category = ""

    def init_end(self):
        self.name = "".join(i.capitalize() for i in self.prefix.split("_"))
        self.opname = self.parent_name + "_" + self.prefix

    def create_op(self, op_type, op_prefix):
        self.op = type(
            self.name,
            (op_type,),
            {
                "bl_idname": op_prefix + "." + self.parent_name + "_" + self.prefix,
                "bl_label": " ".join(self.prefix.split("_")).capitalize(),
                "bl_description": self.info,
                "my_props": self.props.keys(),
                "prefix": self.prefix,
                "parent_name": self.parent_name,
                "payload": self.payload,
            },
        )
        setattr(self.op, "__annotations__", {})
        for k, v in self.props.items():
            self.op.__annotations__[k] = v

    def __init__(self, master_name):
        self.init_begin(master_name)
        self.generate()
        self.init_end()
        self.create_op()
