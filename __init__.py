# ##### BEGIN GPL LICENSE BLOCK #####
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, version 3 of the license.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Ambitools",
    "category": "Mesh",
    "description": "Various tools for mesh processing",
    "author": "ambi",
    "location": "3D view > Tools",
    "version": (1, 1, 0),
    "blender": (2, 79, 0)
}

# import/reload all source files
if "bpy" in locals():
    import importlib
    importlib.reload(au)
else:
    from . import amb_utils as au

import bpy

import numpy as np
import bpy
import bmesh
import random
import cProfile, pstats, io
from collections import defaultdict, OrderedDict


def op_fraggle(mesh, thres, n):
    verts = read_verts(mesh)
    edges = read_edges(mesh)
    edge_a, edge_b = edges[:,0], edges[:,1]
    #for i in range(len(edge_a)):
    #    fastverts[edge_a[i]] += tvec[i]*thres 
    #    fastverts[edge_b[i]] -= tvec[i]*thres  
    for _ in range(n):
        tvec = verts[edge_b] - verts[edge_a]
        #tvlen = np.linalg.norm(tvec, axis=1)
        #tvec = (tvec.T / tvlen).T 
        verts[edge_a] += tvec * thres 
        verts[edge_b] -= tvec * thres 
    write_verts(mesh, verts)


def op_smooth_mask(verts, edges, mask, n):
    #for e in edges:
    #    edge_c[e[0]] += 1
    #    edge_c[e[1]] += 1
    edge_c = np.zeros(len(verts), dtype=np.int32)
    e0_u, e0_c = np.unique(edges[:,0], return_counts=True)
    e1_u, e1_c = np.unique(edges[:,1], return_counts=True)
    edge_c[e0_u] += e0_c
    edge_c[e1_u] += e1_c
    edge_c = edge_c.T

    new_verts = np.copy(verts)
    new_verts = new_verts.T
    mt1 = (1.0-mask).T
    mt0 = mask.T
    for xc in range(n):
        # <new vert location> = sum(<connected locations>) / <number of connected locations>
        locs = np.zeros((len(verts), 3), dtype=np.float64)
        np.add.at(locs, edges[:,0], verts[edges[:,1]])
        np.add.at(locs, edges[:,1], verts[edges[:,0]])

        locs = locs.T
        locs /= edge_c
        locs *= mt1

        new_verts *= mt0
        new_verts += locs 

    return new_verts.T


class Mesh_Operator(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}
    my_props = []
    prefix = ""
    parent_name = ""

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def payload(self, mesh, context):
        pass

    def invoke(self, context, event):
        #self.pr = au.profiling_start()

        # copy property values from panel to operator
        print(self.prefix, self.my_props)
        if self.prefix != "":
            for p in self.my_props:
                opname = self.parent_name + "_" + self.prefix + "_" + p
                setattr(self, p, getattr(context.scene, opname)) 
                print(opname, getattr(context.scene, opname))

        return self.execute(context)

    def execute(self, context):
        # apply modifiers for the active object before mesh actions
        for mod in context.active_object.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError as ex:
                b_print(ex)    

        # run mesh operation
        mesh = context.active_object.data
        self.payload(mesh, context)
        mesh.update(calc_edges=True)

        #au.profiling_end(self.pr)

        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        
        for p in self.my_props:
            row = col.row()
            row.prop(self, p, expand=True)


def mesh_operator_factory(props, prefix, payload, name, parent_name):
    return type('name', (Mesh_Operator,), 
        {**{'bl_idname' : "object." + parent_name + "_" + prefix,
        'bl_label' : " ".join(prefix.split("_")).capitalize(),
        'my_props' : props.keys(),
        'prefix' : prefix,
        'parent_name' : parent_name,
        'payload' : payload
        }, ** props})


class PanelBuilder:
    def __init__(self, master_name, master_panel, mesh_ops):
        self.panel = {i.prefix : bpy.props.BoolProperty(
                name=i.prefix.capitalize() + " settings",
                description="Display settings of the tool",
                default=False) for i in mesh_ops}      

        self.master_name = master_name
        self.master_panel = master_panel
        self.mesh_ops = mesh_ops

    def create_panel(this):
        class _pt(bpy.types.Panel):
            bl_label = this.master_name.capitalize()
            bl_idname = this.master_panel

            bl_space_type = 'VIEW_3D'
            bl_region_type = 'TOOLS'
            bl_category = "Tools"

            def draw(self, context):
                layout = self.layout
                col = layout.column(align=True)

                for mop in this.mesh_ops:
                    split = col.split(percentage=0.15, align=True)
                    opname = this.master_panel + "_" + mop.prefix
                    
                    if len(mop.props) == 0:
                        split.prop(context.scene, opname, text="", icon='LINK')
                    else:
                        if getattr(context.scene, opname):
                            split.prop(context.scene, opname, text="", icon='DOWNARROW_HLT')
                        else:
                            split.prop(context.scene, opname, text="", icon='RIGHTARROW')

                    opr = split.operator(mop.op.bl_idname, text = " ".join(mop.prefix.split("_")).capitalize())

                    if getattr(context.scene, opname):
                        box = col.column(align=True).box().column()
                        for i, p in enumerate(mop.props):
                            if i%2==0:
                                row = box.row(align=True)
                            row.prop(context.scene, this.master_name+"_"+mop.prefix + "_" + p)
        return _pt

    def register_params(self):
        for mesh_op in self.mesh_ops:
            bpy.utils.register_class(mesh_op.op)
            for k, v in mesh_op.props.items():
                setattr(bpy.types.Scene, mesh_op.parent_name+"_"+mesh_op.prefix+"_"+k, v)

        for k, v in self.panel.items():
            setattr(bpy.types.Scene, self.master_panel+"_"+k, v)

    def unregister_params(self):
        for mesh_op in self.mesh_ops:
            bpy.utils.unregister_class(mesh_op.op)
            for k, v in mesh_op.props.items():
                delattr(bpy.types.Scene, mesh_op.parent_name+"_"+mesh_op.prefix+"_"+k)

        for k, v in self.panel.items():
            delattr(bpy.types.Scene, self.master_panel+"_"+k)

class Master_OP:
    def generate(self):
        pass

    def __init__(self):
        self.props = OrderedDict()
        self.parent_name = "ambitools"

        self.generate()

        def _wrap(this, mesh, context):
            mode = bpy.context.object.mode
            bpy.ops.object.mode_set(mode = self.start_mode)
            self.payload(this, mesh, context)
            bpy.ops.object.mode_set(mode = mode)

        self.op = mesh_operator_factory(self.props, self.prefix, _wrap, self.name, self.parent_name)


class Masked_Smooth_OP(Master_OP):
    def generate(self):
        self.props['power']  = bpy.props.FloatProperty(name="Power", default=0.7, min=0.0, max=10.0)
        self.props['iter']   = bpy.props.IntProperty(name="Iterations", default=2, min=1, max=10)
        self.props['border'] = bpy.props.BoolProperty(name="Exclude border", default=True)

        self.prefix = "masked_smooth"
        self.name = "OBJECT_OT_Maskedsmooth"
        self.start_mode = 'OBJECT'

        def _pl(self, mesh, context):
            verts = au.read_verts(mesh)
            edges = au.read_edges(mesh)
            norms = au.read_norms(mesh)

            curve = np.abs(au.calc_curvature(verts, edges, norms)-0.5)
            curve = au.mesh_smooth_filter_variable(curve, verts, edges, 1)
            
            curve -= np.min(curve)
            curve /= np.max(curve)
            curve *= 8.0 * self.power
            curve = np.where(curve>1.0, 1.0, curve)

            # don't move border
            if self.border:
                curve = np.where(au.get_nonmanifold_verts(mesh), 1.0, curve)

            new_verts = op_smooth_mask(verts, edges, curve, self.iter)

            au.write_verts(mesh, new_verts)

        self.payload = _pl


class CropToLarge_OP(Master_OP):
    def generate(self):
        self.props['shells']  = bpy.props.IntProperty(name="Shells", default=1, min=1, max=100)

        self.prefix = "crop_to_large"
        self.name = "OBJECT_OT_CropToLarge"
        self.start_mode = 'EDIT'

        def _pl(self, mesh, context):
            with au.Bmesh_from_edit(mesh) as bm:
                shells = au.mesh_get_edge_connection_shells(bm)
                print(len(shells), "shells")

                for i in range(len(bm.faces)):
                    bm.faces[i].select = True

                delete_this = list(sorted(shells, key=lambda x: -len(x)))[:self.shells]
                for s in delete_this:
                    for f in s:
                        bm.faces[f.index].select = False

            bpy.ops.mesh.delete(type='FACE')
            
            if False:
                pass
                # find separated shells (no edge connections between verts)
                # idx =  np.arange((len(verts)), dtype=np.uint32)
                # parr = np.copy(idx)
                # while True:
                #     np.minimum.at(idx, edges[:,0], idx[edges[:,1]])
                #     np.minimum.at(idx, edges[:,1], idx[edges[:,0]])
                #     # optimize: when a number is overwritten, replace all numbers with new, not just the one
                #     if np.all(idx == parr):
                #         break
                #     else:
                #         parr = np.copy(idx)

                # uniques, counts = np.unique(idx, return_counts=True)
                # print(len(uniques))
                # big_shell = sorted(zip(uniques, counts), key=lambda x: x[1])[-1][0]
                # mesh.vertices.foreach_set("select", idx != big_shell)

                #bpy.ops.object.mode_set(mode = 'EDIT')
                #bpy.ops.mesh.delete(type='VERT')
                #bpy.ops.object.mode_set(mode = 'OBJECT')

                #delete_these =  np.nonzero(idx != big_shell)[0]
                # delete all verts that aren't part of the big shell
                # bm = bmesh.new()
                # bm.from_mesh(mesh)
                # bm.verts.ensure_lookup_table()
                # verts = [bm.verts[i] for i in delete_these]
                # for v in verts:
                #     v.select = True
                # bmesh.ops.delete(bm, geom=verts)
                # bm.to_mesh(mesh)
                # bm.free()

        self.payload = _pl


class Cleanup_OP(Master_OP):
    def generate(self):
        #self.props['trifaces'] = bpy.props.BoolProperty(name="Only trifaces", default=False)
        self.props['fillface'] = bpy.props.BoolProperty(name="Fill faces", default=True)
        self.prefix = "cleanup_triface"
        self.name = "OBJECT_OT_CleanupTriface"
        self.start_mode = 'EDIT'
        
        def _pl(self, mesh, context):
            with au.Bmesh_from_edit(mesh) as bm:
                # deselect all
                for v in bm.verts:
                    v.select = False

                # some preprocessing
                #e_len = np.empty((len(bm.edges)), dtype=np.float32)
                #for e in bm.edges:
                #    e_len[e.index] = (e.verts[0].co - e.verts[1].co).length
                #print(np.min(e_len), np.mean(e_len))
                #bmesh.ops.dissolve_degenerate(bm) #, dist=np.min(e_len))

                # find nonmanifold edges
                nm_edges = np.zeros((len(bm.edges)), dtype=np.bool)
                c3_edges = np.zeros((len(bm.edges)), dtype=np.bool)
                for e in bm.edges:
                    facecount = len(e.link_faces)
                    if facecount < 2:
                        nm_edges[e.index] = True
                    elif facecount > 2:
                        c3_edges[e.index] = True

                # A

                # remove all faces, connected to 3+ connection edge, that have nonmanifold edges
                delete_this = []
                for f in bm.faces:
                    nm = False
                    c3 = False
                    for e in f.edges:
                        if nm_edges[e.index]:
                            nm = True
                        if c3_edges[e.index]:
                            c3 = True
                    if nm and c3:
                        delete_this.append(f)               

                bmesh.ops.delete(bm, geom=delete_this, context=5)

                #if self.trifaces == False:
                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()

                c3_edges = np.zeros((len(bm.edges)), dtype=np.bool)
                for e in bm.edges:
                    if len(e.link_faces) > 2:
                        c3_edges[e.index] = True

                # B

                # mark non manifold verts (3-face-edge)
                # delete verts, select edges around the deleted vertices
                nonm_verts = set([])
                nonm_edges_idx = np.nonzero(c3_edges)[0]
                nonm_edges = [bm.edges[e] for e in nonm_edges_idx]

                for e in nonm_edges:
                    e.select = True
                    nonm_verts.add(e.verts[0].index)
                    nonm_verts.add(e.verts[1].index)

                for v in nonm_verts:
                    for v in au.vert_vert(bm.verts[v]):
                        v.select = True

                # enum {
                # DEL_VERTS = 1,
                # DEL_EDGES,
                # DEL_ONLYFACES,
                # DEL_EDGESFACES,
                # DEL_FACES,
                # DEL_ALL,
                # DEL_ONLYTAGGED
                # };

                delete_this = [bm.verts[v] for v in nonm_verts]
                bmesh.ops.delete(bm, geom=delete_this, context=1)

                # delete loose edges
                bm.edges.ensure_lookup_table()
                loose_edges = []
                for e in bm.edges:
                    if len(e.link_faces) == 0:
                        loose_edges.append(e)
                bmesh.ops.delete(bm, geom=loose_edges, context=2)

                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()

                for e in bm.edges:
                    if len(e.link_faces) > 1 or len(e.link_faces) == 0:
                        e.select = False

                # C

                # fill faces for each loop
                # triangulate
                if self.fillface:
                    loops = au.bmesh_get_boundary_edgeloops_from_selected(bm)
                    au.bmesh_deselect_all(bm)

                    # TODO: filter out invalid loops(?)
                    new_faces = []
                    leftover_loops = []

                    for l in loops:
                        nl = au.bmesh_vertloop_from_edges(l)
                        if nl:
                            f = bm.faces.new(nl)
                            f.select = True
                            f.smooth = True
                            new_faces.append(f)
                        else:
                            leftover_loops.append(l)

                    au.bmesh_deselect_all(bm)
                    for l in leftover_loops:
                        for e in l:
                            e.select = True

                    loops = au.bmesh_get_boundary_edgeloops_from_selected(bm)

                    for l in loops:
                        nl = au.bmesh_vertloop_from_edges(l)
                        if nl:
                            f = bm.faces.new(nl)
                            f.select = True
                            f.smooth = True
                            new_faces.append(f)

                    for f in new_faces:
                        f.select = True

                    bmesh.ops.recalc_face_normals(bm, faces=new_faces)
                    res = bmesh.ops.triangulate(bm, faces=new_faces)
                    smooth_verts = []
                    for f in res['faces']:
                        for v in f.verts:
                            smooth_verts.append(v)
                    smooth_verts = list(set(smooth_verts))
                    print(len(smooth_verts), "smoothed verts")
                    bmesh.ops.smooth_vert(bm, verts=smooth_verts, factor=1.0, use_axis_x=True, use_axis_y=True, use_axis_z=True)

                    # cleanup faces with no other face connections
                    bm.faces.ensure_lookup_table()
                    delete_this = []
                    for f in bm.faces:
                        no_conn = True
                        for e in f.edges:
                            if e.is_manifold:
                                no_conn = False
                        if no_conn:
                            delete_this.append(f)     

                    print(len(delete_this), "faces deleted after triface cleanup")
                    bmesh.ops.delete(bm, geom=delete_this, context=5)


        self.payload = _pl


pbuild = PanelBuilder("ambitools", "ambitools_panel", [Masked_Smooth_OP(), CropToLarge_OP(), Cleanup_OP()])
OBJECT_PT_ToolsAMB = pbuild.create_panel()

def register():
    pbuild.register_params()
    bpy.utils.register_class(OBJECT_PT_ToolsAMB)

def unregister():
    pbuild.unregister_params()
    bpy.utils.unregister_class(OBJECT_PT_ToolsAMB)
    

