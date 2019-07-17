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

# import numba
# import time


bl_info = {
    "name": "Mesh Refine Toolbox",
    "category": "Mesh",
    "description": "Various tools for mesh processing",
    "author": "ambi",
    "location": "3D view > Tools",
    "version": (1, 1, 11),
    "blender": (2, 80, 0),
}


print("Import: __init__.py")

import bpy  # noqa:F401
import numpy as np  # noqa:F401
import bmesh  # noqa:F401
from collections import OrderedDict  # noqa:F401
import mathutils as mu  # noqa:F401

# import/reload all source files
if "afm" not in locals():
    print("amb_tools: Import")
    from .bpy_amb import utils as au  # noqa:F401
    from .bpy_amb import fastmesh as afm  # noqa:F401
    from .bpy_amb import bbmesh as abm  # noqa:F401
    from .bpy_amb import raycast  # noqa:F401
    from .bpy_amb import vcol
    from . import mesh_ops  # noqa:F401
else:
    import importlib

    print("amb_tools: Reload")
    importlib.reload(au)
    importlib.reload(afm)
    importlib.reload(abm)
    importlib.reload(raycast)
    importlib.reload(mesh_ops)
    importlib.reload(vcol)

from collections import defaultdict


class PanelBuilder:
    def __init__(self, master_name, input_ops):
        mesh_ops = [i(master_name) for i in input_ops]

        # inject panel functionality into operators
        # def _inject(cl):
        #     def invoke(self, context, event):
        #         # copy property values from panel to operator
        #         if self.prefix != "":
        #             for p in self.my_props:
        #                 opname = self.parent_name + "_" + self.prefix + "_" + p
        #                 print(opname)
        #                 panel_value = getattr(context.scene, opname)
        #                 setattr(self, p, panel_value)
        #         return self.execute(context)

        #     def draw(self, context):
        #         layout = self.layout
        #         col = layout.column()

        #         for p in self.my_props:
        #             row = col.row()
        #             row.prop(self, p, expand=True)

        #     cl.draw = draw
        #     cl.invoke = invoke

        # for m in mesh_ops:
        #     _inject(m)

        # ---
        self.panel = {
            i.prefix: bpy.props.BoolProperty(
                name=i.prefix.capitalize() + " settings",
                description="Display settings of the tool",
                default=False,
            )
            for i in mesh_ops
        }

        self.categories = set()
        self.draw_order = defaultdict(list)
        for i in mesh_ops:
            self.categories.add(i.category)
            self.draw_order[i.category].append(i)

        self.master_name = master_name
        self.mesh_ops = mesh_ops

    def create_panel(this):
        class _pt(bpy.types.Panel):
            bl_label = " ".join([i.capitalize() for i in this.master_name.split("_")])
            bl_idname = (
                "OBUILD_PT_"
                + "".join([i.capitalize() for i in this.master_name.split("_")])
                + "_panel"
            )

            bl_space_type = "VIEW_3D"
            bl_region_type = "UI"
            bl_category = "Tools"

            def draw(self, context):
                layout = self.layout
                mcol = layout

                for cat in this.draw_order.keys():
                    col = mcol.box().column(align=True)
                    col.label(text=cat)
                    for mop in this.draw_order[cat]:
                        split = col.split(factor=0.15, align=True)
                        opname = this.master_name + "_panel_" + mop.prefix

                        if len(mop.props) == 0:
                            split.prop(context.scene, opname, text="", icon="DOT")
                        else:
                            if getattr(context.scene, opname):
                                split.prop(context.scene, opname, text="", icon="DOWNARROW_HLT")
                            else:
                                split.prop(context.scene, opname, text="", icon="RIGHTARROW")

                        split.operator(
                            mop.op.bl_idname, text=" ".join(mop.prefix.split("_")).capitalize()
                        )

                        if getattr(context.scene, opname):
                            box = col.column(align=True).box().column()
                            for i, p in enumerate(mop.props):
                                # if i % 2 == 0:
                                #     row = box.row(align=True)
                                row = box.row(align=True)
                                row.prop(context.scene, mop.opname + "_" + p)

        return _pt

    def register_params(self):
        for mesh_op in self.mesh_ops:
            bpy.utils.register_class(mesh_op.op)
            for k, v in mesh_op.props.items():
                setattr(bpy.types.Scene, mesh_op.opname + "_" + k, v)

        for k, v in self.panel.items():
            setattr(bpy.types.Scene, self.master_name + "_panel_" + k, v)

    def unregister_params(self):
        for mesh_op in self.mesh_ops:
            bpy.utils.unregister_class(mesh_op.op)
            for k, _ in mesh_op.props.items():
                delattr(bpy.types.Scene, mesh_op.opname + "_" + k)

        for k, _ in self.panel.items():
            delattr(bpy.types.Scene, self.master_name + "_panel_" + k)


class MaskedSmooth_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["power"] = bpy.props.FloatProperty(name="Power", default=0.5, min=0.0, max=10.0)
        self.props["cutoff"] = bpy.props.FloatProperty(name="Cutoff", default=0.9, min=0.0, max=1.0)
        self.props["iter"] = bpy.props.IntProperty(name="Iterations", default=2, min=1, max=100)
        self.props["blur"] = bpy.props.IntProperty(name="Blur", default=2, min=0, max=10)
        self.props["border"] = bpy.props.BoolProperty(name="Exclude border", default=True)
        self.props["invert"] = bpy.props.BoolProperty(name="Invert curve influence", default=False)

        self.prefix = "masked_smooth"
        self.fastmesh = True
        self.category = "Filter"

        def _pl(self, mesh, context):
            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)
            norms = afm.read_norms(mesh)
            non_manifold = abm.get_nonmanifold_verts(mesh)

            for _ in range(self.iter):
                curve = np.abs(afm.calc_curvature(verts, edges, norms) - 0.5)
                curve = afm.mesh_smooth_filter_variable(curve, verts, edges, self.blur)

                curve -= np.min(curve)
                curve /= np.max(curve)
                curve *= 8.0 * self.power

                curve = np.where(curve > self.cutoff, 1.0, curve)

                if self.invert:
                    curve = 1.0 - curve

                # don't move border
                if self.border:
                    curve = np.where(non_manifold, 1.0, curve)

                verts = afm.op_smooth_mask(verts, edges, curve, 1)

            afm.write_verts(mesh, verts)

            mesh.update(calc_edges=True)

        self.payload = _pl


class CropToLarge_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["shells"] = bpy.props.IntProperty(name="Shells", default=1, min=1, max=100)

        self.prefix = "crop_to_large"
        self.info = "Removes all tiniest disconnected pieces up to specified amount"

        self.category = "Cleanup"

        def _pl(self, bm, context):
            shells = abm.mesh_get_edge_connection_shells(bm)
            print(len(shells), "shells")

            for i in range(len(bm.faces)):
                bm.faces[i].select = True

            # delete_this = list(sorted(shells, key=lambda x: -len(x)))[:self.shells]
            # for s in delete_this:
            #     for f in s:
            #         bm.faces[f.index].select = False

            sorted_shells = list(sorted(shells, key=lambda x: len(x)))
            selected_faces = []
            for s in sorted_shells[self.shells :]:
                for f in s:
                    selected_faces.append(bm.faces[f.index])

            bmesh.ops.delete(bm, geom=list(set(selected_faces)), context="FACES")
            # bpy.ops.mesh.delete(type='FACE')

        self.payload = _pl


class MergeTiny_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["threshold"] = bpy.props.FloatProperty(
            name="Threshold", default=0.02, min=0.0, max=1.0
        )

        self.prefix = "merge_tiny_faces"
        self.info = "Collapse faces with smaller perimeter than defined"

        self.category = "Cleanup"

        def _pl(self, bm, context):
            # thin faces
            collapse_these = []
            avg = sum(f.calc_perimeter() for f in bm.faces) / len(bm.faces)
            for f in bm.faces:
                if f.calc_perimeter() < avg * self.threshold:
                    collapse_these.extend(f.edges)

            bmesh.ops.collapse(bm, edges=list(set(collapse_these)))
            bmesh.ops.connect_verts_concave(bm, faces=bm.faces)

        self.payload = _pl


class EvenEdges_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["amount"] = bpy.props.FloatProperty(name="Amount", default=1.0, min=0.0, max=1.0)
        self.props["iterations"] = bpy.props.IntProperty(
            name="Iterations", default=1, min=1, max=20
        )

        self.prefix = "make_even_edges"
        self.info = "Attempts to equalize edge lengths"

        self.category = "Filter"

        def _pl(self, bm, context):
            avg = sum(e.calc_length() for e in bm.edges) / len(bm.edges)
            for _ in range(self.iterations):
                for e in bm.edges:
                    grow = (avg - e.calc_length()) / 2 * self.amount
                    center = (e.verts[0].co + e.verts[1].co) / 2
                    e.verts[1].co += (e.verts[1].co - center).normalized() * grow
                    e.verts[0].co += (e.verts[0].co - center).normalized() * grow

        self.payload = _pl


class SurfaceSmooth_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["border"] = bpy.props.BoolProperty(name="Exclude border", default=True)
        self.props["iter"] = bpy.props.IntProperty(name="Iterations", default=2, min=1, max=10)

        self.prefix = "surface_smooth"
        self.info = "Smoothing along mesh surface"

        self.category = "Filter"

        def _pl(self, bm, context):
            limit_verts = set([])
            if self.border:
                for e in bm.edges:
                    if len(e.link_faces) < 2:
                        limit_verts.add(e.verts[0].index)
                        limit_verts.add(e.verts[1].index)

            for _ in range(self.iter):
                for v in bm.verts:
                    if v.index in limit_verts:
                        continue

                    ring1 = abm.vert_vert(v)
                    projected = []
                    for rv in ring1:
                        nv = rv.co - v.co
                        dist = nv.dot(v.normal)
                        projected.append(rv.co - dist * v.normal)

                    new_loc = mu.Vector([0.0, 0.0, 0.0])
                    for p in projected:
                        new_loc += p
                    new_loc /= len(projected)

                    v.co = new_loc

        self.payload = _pl


class FacePush_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["object"] = bpy.props.PointerProperty(name="Wrap target", type=bpy.types.Object)
        self.props["distance"] = bpy.props.FloatProperty(name="Distance", min=0.0, default=1.0)

        self.prefix = "face_push"
        self.info = "Push faces to match a surface"

        self.category = "Filter"

        def _pl(self, bm, context):
            for f in bm.faces:
                fm = f.calc_center_median()
                c = self.object.ray_cast(fm, f.normal, distance=self.distance)
                if c[0]:
                    norm = c[1] - fm

                    # calc vert norms
                    locs = []
                    for v in f.verts:
                        r = self.object.ray_cast(v.co, -f.normal, distance=self.distance)
                        if r[0]:
                            # above surface
                            nv = r[1] - v.co
                            # didn't hit a back face
                            if nv.dot(-f.normal) < 0:
                                locs.append(nv)
                        else:
                            # below surface
                            r2 = self.object.ray_cast(v.co, f.normal, distance=self.distance)
                            if r2[0]:
                                nv = r2[1] - v.co
                                # didn't hit a front face
                                if nv.dot(f.normal) > 0:
                                    locs.append(nv)

                    total = mu.Vector((0, 0, 0))

                    if len(locs) > 0:
                        # average of face normals projected from verts
                        for l in locs:
                            total += l
                        total /= len(locs)

                    total = (total + norm) / 2

                    # move verts
                    for v in f.verts:
                        v.co += total

        self.payload = _pl


class EdgeSmooth_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["border"] = bpy.props.BoolProperty(name="Exclude border", default=True)
        self.props["iter"] = bpy.props.IntProperty(name="Iterations", default=2, min=1, max=10)
        self.props["thres"] = bpy.props.FloatProperty(
            name="Threshold", default=0.95, min=0.0, max=1.0
        )

        self.prefix = "edge_smooth"
        self.category = "Filter"

        def _pl(self, bm, context):
            limit_verts = set([])
            if self.border:
                for e in bm.edges:
                    if len(e.link_faces) < 2:
                        limit_verts.add(e.verts[0].index)
                        limit_verts.add(e.verts[1].index)

            # record initial normal field
            normals = []
            for v in bm.verts:
                normals.append(v.normal)

            thr = self.thres

            for _ in range(self.iter):
                # project surrounding verts to normal plane and move <v> to center
                new_verts = []
                for v in bm.verts:
                    new_verts.append(v.co)
                    if v.index in limit_verts:
                        continue

                    v_norm = normals[v.index]
                    ring1 = abm.vert_vert(v)

                    # get projected points on plane defined by v_norm
                    projected = []
                    n_diff = []
                    for rv in ring1:
                        nv = rv.co - v.co
                        dist = nv.dot(v_norm)
                        projected.append(rv.co - dist * v_norm)
                        n_diff.append(rv.co - projected[-1])

                    # get approximate co-planar verts
                    coplanar = []
                    discord = []
                    for i, rv in enumerate(ring1):
                        r_norm = normals[rv.index]
                        if r_norm.dot(v_norm) > thr:
                            coplanar.append((i, rv))
                        else:
                            discord.append((i, rv))

                    for i, rv in discord:
                        # project 2-plane intersection instead of location
                        # which direction is the point? (on the v.normal plane)
                        # make it a 1.0 length vector
                        p = projected[i]
                        p = (p - v.co).normalized()

                        # v + n*p = <the normal plane of rv>, find n
                        d = r_norm.dot(p)
                        # if abs(d) > 1e-6:
                        if d > 1e-6:
                            w = v.co - rv.co
                            fac = r_norm.dot(w) / d
                            u = p * fac

                            # sanity limit for movement length
                            # this doesn't actually prevent the explosion
                            # just makes it a little more pleasing to look at
                            dist = v.co - rv.co
                            if u.length > dist.length:
                                u = u * dist.length / u.length

                            projected[i] = v.co + u
                            # projected = [v.co + u]
                            break
                        else:
                            projected[i] = v.co

                    final_norm = v_norm
                    for i, rv in coplanar:
                        final_norm += r_norm

                    normals[v.index] = final_norm.normalized()

                    if len(projected) > 0:
                        new_loc = mu.Vector([0.0, 0.0, 0.0])
                        for p in projected:
                            new_loc += p
                        new_verts[-1] = new_loc / len(projected)

                    # move towards average valid 1-ring plane
                    # TODO: this should project to new normal (from coplanar norms),
                    # not old v.normal
                    # new_verts[-1] = v.co
                    # if len(coplanar) > 0:
                    #     total = mu.Vector([0.0, 0.0, 0.0])
                    #     for i, rv in coplanar:
                    #         total += n_diff[i]
                    #     total /= len(coplanar)
                    #     new_verts[-1] += total

                # finally set new values for verts
                for i, v in enumerate(bm.verts):
                    v.co = new_verts[i]

            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

            # mesh.update(calc_edges=True)

        self.payload = _pl


class Mechanize_OP(mesh_ops.MeshOperatorGenerator):
    # @numba.jit
    def _nb(self, iverts, ring1s, norms):
        for v_i in range(iverts.shape[0]):
            v = iverts[v_i]
            ring1 = np.array(ring1s[v_i])
            projected = []
            distances = np.empty((len(ring1),))
            for rv_i, rv in enumerate(ring1):
                nv = rv - v
                dist = nv.dot(norms[v_i])
                distances[rv_i] = abs(dist) / np.linalg.norm(nv)
                projected.append(rv - dist * norms[v_i])

            # nv = ring1 - v
            # dist = nv.dot(norms[v_i])
            # distances = np.abs(dist) / np.linalg.norm(nv)
            # projected = ring1 - np.array([d * norms[v_i] for d in dist])

            dist_sum = np.sum(distances)
            new_loc = np.zeros((3,))

            if dist_sum / len(projected) > 0.02:
                for i, p in enumerate(projected):
                    new_loc += p * distances[i] / dist_sum
            else:
                for i, p in enumerate(projected):
                    new_loc += p
                new_loc /= len(projected)

            iverts[v_i] = new_loc

    def generate(self):

        """
        1. find difference vectors (D) to one ring (R)
        2. divide angle between vertex normal (N) and difference vector (A) with
           diff vec length (D.len)
        3. projected vectors (P) = R - D * N
        4. new vertex location (V) = V + P[i] * distances[i] / sum(distances)
        """

        self.props["border"] = bpy.props.BoolProperty(name="Exclude border", default=True)
        self.props["iter"] = bpy.props.IntProperty(name="Iterations", default=2, min=1, max=50)

        self.prefix = "mechanize"
        self.info = "Artistic mesh processing, going for a chiseled look"

        self.category = "Filter"

        def _pl(self, bm, context):
            limit_verts = set([])
            if self.border:
                for e in bm.edges:
                    if len(e.link_faces) < 2:
                        limit_verts.add(e.verts[0].index)
                        limit_verts.add(e.verts[1].index)

            ok_verts = []
            for v in bm.verts:
                if v.index not in limit_verts:
                    ok_verts.append(v)

            ring1s = []
            for v in bm.verts:
                ring1s.append(abm.vert_vert(v))

            for xx in range(self.iter):
                print("iteration:", xx + 1)
                for v in ok_verts:
                    ring1 = ring1s[v.index]
                    projected = []
                    distances = []
                    for rv in ring1:
                        nv = rv.co - v.co
                        dist = nv.dot(v.normal)
                        distances.append(abs(dist) / nv.length)
                        projected.append(rv.co - dist * v.normal)

                    dist_sum = sum(distances)
                    new_loc = mu.Vector([0.0, 0.0, 0.0])

                    if dist_sum / len(projected) > 0.02:
                        for i, p in enumerate(projected):
                            new_loc += p * distances[i] / dist_sum
                    else:
                        for i, p in enumerate(projected):
                            new_loc += p
                        new_loc /= len(projected)

                    v.co = new_loc

        self.payload = _pl


class CleanupThinFace_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["threshold"] = bpy.props.FloatProperty(
            name="Threshold", default=0.95, min=0.0, max=1.0
        )
        self.props["repeat"] = bpy.props.IntProperty(name="Repeat", default=2, min=0, max=10)

        self.prefix = "cleanup_thin_faces"
        self.info = "Collapse thin faces"

        self.category = "Cleanup"

        def _pl(self, bm, context):
            thr = self.threshold

            for _ in range(self.repeat):
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

                collapse_these = []
                for f in bm.faces:
                    s = 0.0
                    for e in f.edges:
                        s += e.calc_length()
                    s = s / 2 * thr
                    for e in f.edges:
                        if e.calc_length() > s:
                            mval = 100000.0
                            sed = None
                            for e in f.edges:
                                cl = e.calc_length()
                                if cl < mval:
                                    mval = cl
                                    sed = e
                            collapse_these.append(sed)
                            break

                # cthese = [bm.faces[i].edges[j] for i, j in res]
                cthese = list(set(collapse_these))
                print(len(cthese), "collapsed edges")

                bmesh.ops.collapse(bm, edges=cthese)
                bmesh.ops.connect_verts_concave(bm, faces=bm.faces)

        self.payload = _pl


class Cleanup_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        # self.props['trifaces'] = bpy.props.BoolProperty(name="Only trifaces", default=False)
        self.props["fillface"] = bpy.props.BoolProperty(name="Fill faces", default=True)
        self.prefix = "cleanup_triface"
        self.info = (
            "Removes all edges with more than two faces and tries to rebuild the surrounding mesh"
        )

        self.category = "Cleanup"

        def _pl(self, bm, context):
            # deselect all
            for v in bm.verts:
                v.select = False

            # some preprocessing
            # e_len = np.empty((len(bm.edges)), dtype=np.float32)
            # for e in bm.edges:
            #    e_len[e.index] = (e.verts[0].co - e.verts[1].co).length
            # print(np.min(e_len), np.mean(e_len))
            # bmesh.ops.dissolve_degenerate(bm) #, dist=np.min(e_len))

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

            bmesh.ops.delete(bm, geom=delete_this, context="FACES")

            # if self.trifaces == False:
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
                for v in abm.vert_vert(bm.verts[v]):
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
            bmesh.ops.delete(bm, geom=delete_this, context="VERTS")

            # delete loose edges
            bm.edges.ensure_lookup_table()
            loose_edges = []
            for e in bm.edges:
                if len(e.link_faces) == 0:
                    loose_edges.append(e)
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")

            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()

            for e in bm.edges:
                if len(e.link_faces) > 1 or len(e.link_faces) == 0:
                    e.select = False

            # C

            # fill faces for each loop
            # triangulate
            if self.fillface:
                all_faces = []
                for _ in range(2):
                    bm.edges.ensure_lookup_table()
                    loops = abm.bmesh_get_boundary_edgeloops_from_selected(bm)
                    new_faces, leftover_loops = abm.bmesh_fill_from_loops(bm, loops)

                    all_faces.extend(new_faces)
                    abm.bmesh_deselect_all(bm)

                    for l in leftover_loops:
                        for e in l:
                            e.select = True

                    # TODO: loops with 4 edge connections (one vert) could be
                    #       split into 2 verts which makes the loops simple

                    print(len(leftover_loops))
                    if len(leftover_loops) == 0:
                        break

                for f in all_faces:
                    f.select = True

                bmesh.ops.recalc_face_normals(bm, faces=all_faces)
                res = bmesh.ops.triangulate(bm, faces=all_faces)
                smooth_verts = []
                for f in res["faces"]:
                    for v in f.verts:
                        smooth_verts.append(v)
                smooth_verts = list(set(smooth_verts))
                print(len(smooth_verts), "smoothed verts")
                bmesh.ops.smooth_vert(
                    bm,
                    verts=smooth_verts,
                    factor=1.0,
                    use_axis_x=True,
                    use_axis_y=True,
                    use_axis_z=True,
                )

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
                bmesh.ops.delete(bm, geom=delete_this, context="FACES")

        self.payload = _pl


class MeshNoise_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["amount"] = bpy.props.FloatProperty(name="Amount", default=1.0)
        self.props["scale"] = bpy.props.FloatProperty(name="Scale", default=1.0, min=0.0)
        self.props["noisetype"] = bpy.props.EnumProperty(
            items=[
                ("DISTANCE", "Distance", "", "", 0),
                ("CHEBYCHEV", "Chebychev", "", "", 1),
                ("MANHATTAN", "Manhattan", "", "", 2),
            ],
            name="Type",
            default="CHEBYCHEV",
        )
        self.props["noisef"] = bpy.props.EnumProperty(
            items=[("21", "F2-F1", "", "", 0), ("1", "F1", "", "", 1), ("2", "F2", "", "", 2)],
            name="Feature",
            default="21",
        )

        self.prefix = "mesh_noise"
        self.info = "Various noise functions that can be instantly applied on the mesh"

        self.category = "Filter"

        def _pl(self, bm, context):
            df = None
            if self.noisef == "21":
                df = lambda x: x[1] - x[0]
            if self.noisef == "1":
                df = lambda x: x[0]
            if self.noisef == "2":
                df = lambda x: x[1]
            if df is None:
                df = lambda x: x[0]

            for v in bm.verts:
                d, _ = mu.noise.voronoi(
                    v.co * self.scale, distance_metric=self.noisetype, exponent=2.5
                )
                v.co += v.normal * df(d) * self.amount

        self.payload = _pl


class EdgesToCurve_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["balance"] = bpy.props.FloatProperty(
            name="Balance", default=0.99, min=0.0, max=1.0
        )

        self.prefix = "edges_to_curve"
        self.info = "Rotates edges to find optimal local curvature description"

        self.category = "Refine"

        def _pl(self, bm, context):
            traversed = np.zeros((len(bm.edges)), dtype=np.bool)
            for e in bm.edges:
                if traversed[e.index]:
                    continue

                f = e.link_faces
                # only if edge has two faces connected
                if len(f) == 2:
                    # mark all both faces edges as traversed
                    for n in range(2):
                        for i in f[n].edges:
                            traversed[i.index] = True

                    e_len = e.calc_length()

                    # whats the max number edge can be rotated on the 2-face plane
                    max_rots = min(len(f[0].edges) - 2, len(f[1].edges) - 2)

                    # initial fit (find lowest angle between edge vert normals)
                    # vs. edge length
                    v0 = e.verts[0]
                    v1 = e.verts[1]
                    diff_01 = v0.co - v1.co
                    if diff_01.length == 0:
                        continue
                    best = e_len * ((v0.normal.dot(v1.normal) + 1) / 2) * self.balance + (
                        e_len / (diff_01).length
                    ) * (1.0 - self.balance)
                    rotations = 0

                    # select vert that from which you take the next step doesn't end
                    # on a vert on the edge <e> (for each face loop)
                    # so that we have the first rotated edge position

                    fvi = [0, 0]
                    for n in range(2):
                        fvi[n] = [i for i, fv in enumerate(f[n].verts) if e.verts[0] == fv][0]
                        n_step = (fvi[n] + 1) % len(f[n].verts)
                        if f[n].verts[n_step] == e.verts[1]:
                            fvi[n] = n_step

                    for r in range(max_rots):
                        fvi[0] = (fvi[0] + 1) % len(f[0].verts)
                        fvi[1] = (fvi[1] + 1) % len(f[1].verts)

                        v0 = f[0].verts[fvi[0]]
                        v1 = f[1].verts[fvi[1]]

                        if v0 == v1:
                            continue

                        new_calc = e_len * ((v0.normal.dot(v1.normal) + 1) / 2) * self.balance + (
                            e_len / (v0.co - v1.co).length
                        ) * (1.0 - self.balance)

                        if new_calc > best:
                            best = new_calc
                            rotations = r + 1

                    # flip edge to optimal location
                    te = e
                    for _ in range(rotations):
                        te = bmesh.utils.edge_rotate(te, True)
                        if te is None:
                            break

        self.payload = _pl


class SplitQuads_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["thres"] = bpy.props.FloatProperty(
            name="Threshold", default=0.5, min=0.0, max=1.0
        )
        self.props["normals"] = bpy.props.BoolProperty(name="Use Normals", default=False)

        self.prefix = "split_quads"
        self.info = "Triangulates quads, using thresholds and angles"

        self.category = "Refine"

        def _pl(self, bm, context):
            for f in bm.faces:
                # for all quads
                if len(f.edges) == 4:
                    # quad:
                    #  0
                    # 3 1
                    #  2

                    v = [i for i in f.verts]

                    # get two possible cut configurations
                    # either cut v[0],v[2] or v[1],v[3]

                    if not self.normals:
                        # case: v[0],v[2]
                        vec10 = v[0].co - v[1].co
                        vec12 = v[2].co - v[1].co
                        # note: ccw cross product
                        crp102 = vec10.normalized().cross(vec12.normalized())

                        vec30 = v[0].co - v[3].co
                        vec32 = v[2].co - v[3].co
                        crp302 = vec32.normalized().cross(vec30.normalized())

                        case02 = crp102.dot(crp302)

                        # case: v[1],v[3]
                        vec01 = v[1].co - v[0].co
                        vec03 = v[3].co - v[0].co
                        crp013 = vec01.normalized().cross(vec03.normalized())

                        vec21 = v[1].co - v[2].co
                        vec23 = v[3].co - v[2].co
                        crp213 = vec21.normalized().cross(vec23.normalized())

                        case13 = crp013.dot(crp213)
                    else:
                        case02 = v[0].normal.dot(v[2].normal)
                        case13 = v[1].normal.dot(v[3].normal)

                    if abs(case02) < self.thres or abs(case13) < self.thres:
                        if abs(case02) > abs(case13):
                            bmesh.utils.face_split(f, v[0], v[2])
                        else:
                            bmesh.utils.face_split(f, v[1], v[3])

        self.payload = _pl


class RebuildQuads_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["decimate"] = bpy.props.FloatProperty(
            name="Decimate", default=0.1, min=0.0, max=1.0
        )
        # self.props['subdivide'] = bpy.props.IntProperty(name="Subdivide",
        # default=1, min=1, max=10)
        # self.props['quadstep'] = bpy.props.FloatProperty(name="Quad Angle",
        # default=4.0, min=0.0, max=4.0)
        # self.props['smooth'] = bpy.props.BoolProperty(name="Smooth", default=True)

        self.prefix = "rebuild_quads"
        self.info = (
            "Rebuilds mesh with quads by first decimating and making quads from triangles.\n"
            "Then subdividing and projecting to surface with shrinkwrap"
        )

        self.category = "Refine"

        def _pl(self, bm, context):
            indices = [v.index for v in bm.verts]

            bpy.ops.object.mode_set(mode="OBJECT")
            ob = context.object

            temp_object = ob.copy()
            temp_object.data = ob.data.copy()
            temp_object.animation_data_clear()

            print("Curvature.")

            # decimate A
            # m_decimate = ob.modifiers.new(name="Decimate", type="DECIMATE")
            # m_decimate.ratio = self.decimate

            # bpy.ops.object.modifier_apply(modifier=m_decimate.name)

            # decimate B
            mesh = ob.data

            vg = None
            if "curve" not in ob.vertex_groups:
                vg = ob.vertex_groups.new(name="curve")
            else:
                vg = ob.vertex_groups["curve"]

            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)
            norms = afm.read_norms(mesh)

            curve = afm.calc_curvature(verts, edges, norms) - 0.5
            curve = afm.mesh_smooth_filter_variable(curve, verts, edges, 10)
            curve = np.abs(curve)

            # curve -= np.min(curve)
            curve /= np.max(curve)
            curve **= 2.0
            # curve *= 8.0 * self.power

            bpy.ops.object.mode_set(mode="OBJECT")

            for v in indices:
                vg.add([v], curve[v], "REPLACE")

            print("Decimate.")

            m_decimate = ob.modifiers.new(name="Decimate", type="DECIMATE")
            m_decimate.ratio = self.decimate
            m_decimate.vertex_group = vg.name
            m_decimate.vertex_group_factor = 10.0
            m_decimate.invert_vertex_group = True

            bpy.ops.object.modifier_apply(modifier=m_decimate.name)

            # if self.smooth:
            #     bpy.ops.object.mesh_refine_toolbox_surface_smooth(border=True, iter=2)

            print("Quads.")

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            # bpy.ops.mesh.tris_convert_to_quads(
            # face_threshold=self.quadstep, shape_threshold=self.quadstep)
            bpy.ops.mesh.tris_convert_to_quads(face_threshold=3.14159, shape_threshold=3.14159)
            # bpy.ops.object.mesh_refine_toolbox_split_quads(thres=1.0, normals=True)
            bpy.ops.object.mode_set(mode="OBJECT")

            _ = ob.modifiers.new(name="Subd", type="SUBSURF")
            m_swrp = ob.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
            # m_swrp.wrap_method = 'PROJECT'
            # m_swrp.cull_face = 'FRONT'
            m_swrp.target = temp_object

            for mod in context.object.modifiers:
                bpy.ops.object.modifier_apply(modifier=mod.name)

            meshname = temp_object.data.name
            objs = bpy.data.objects
            objs.remove(objs[temp_object.name], do_unlink=True)
            meshes = bpy.data.meshes
            meshes.remove(meshes[meshname], do_unlink=True)
            bpy.ops.object.mode_set(mode="EDIT")

        self.payload = _pl


class RemoveTwoBorder_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.prefix = "remove_two_border"
        self.info = "Removes all faces that have more than two non-manifold borders"

        self.category = "Cleanup"

        def _pl(self, bm, context):
            selected_faces = []
            for f in bm.faces:
                previous = None
                count = 0
                for e in f.edges:
                    if previous is None:
                        previous = e.is_manifold
                    if e.is_manifold != previous:
                        count += 1
                        previous = e.is_manifold

                # more than one shared non-manifold border
                if count > 2:
                    selected_faces.append(f)

            if len(selected_faces) > 0:
                bmesh.ops.delete(bm, geom=selected_faces, context="FACES")

        self.payload = _pl


class CurveSubd_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.prefix = "curve_subd"
        self.info = "Subdivide according to curvature maintaining the position of original points"

        self.category = "Refine"

        def _pl(self, bm, context):
            orig_verts = [v.index for v in bm.verts]

            a, b, geo = bmesh.ops.subdivide_edges(
                # quad_corner_type = ('STRAIGHT_CUT', 'INNER_VERT', 'PATH', 'FAN')
                bm,
                edges=bm.edges,
                smooth=0.5,
                cuts=1,
                use_grid_fill=True,
                use_only_quads=True,
            )

            for v in bm.verts:
                v.select = False

            # make ngons into quads
            for f in bm.faces:
                if len(f.verts) > 4:
                    verts = []
                    for v in f.verts:
                        if v.index not in orig_verts:
                            verts.append(v)

                    # new_loc = mu.Vector([0, 0, 0])
                    # for i in range(len(verts)):
                    #     new_loc += verts[i].co
                    # new_loc /= len(verts)

                    # bmesh.ops.delete(bm, geom=[f], context="FACES_ONLY")

                    r = bmesh.ops.poke(bm, faces=[f], center_mode="MEAN")
                    # , offset=0.0, use_relative_offset=True)

                    dis = set()
                    for nf in r["faces"]:
                        for e in nf.edges:
                            if e.verts[0] not in verts and e.verts[1] not in verts:
                                dis.add(e)

                    bmesh.ops.dissolve_edges(bm, edges=list(dis))

                    # bmesh.ops.connect_verts(bm, verts=verts)
                    # bmesh.ops.triangulate(bm, faces=[f], ngon_method="BEAUTY")
                    # bmesh.ops.unsubdivide(bm, verts=verts, iterations=2)

            # for f in bm.faces:
            #     if len(f.verts) > 4:
            #         for v in f.verts:
            #             if v.index not in orig_verts:
            #                 v.select = True
            #             else:
            #                 v.select = False
            #     else:
            #         for v in f.verts:
            #             v.select = False

        self.payload = _pl


class CurveDecimate_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["thres"] = bpy.props.FloatProperty(
            name="Threshold", default=0.1, min=0.0, max=1.0
        )

        self.props["factor"] = bpy.props.FloatProperty(name="Factor", default=0.1, min=0.0, max=1.0)

        self.prefix = "curve_decimate"
        self.info = "Decimate according to curvature"

        self.category = "Refine"

        def _pl(self, bm, context):
            # for i in range(5):
            #     bmesh.ops.dissolve_limit(bm, angle_limit=i / 10, edges=bm.edges)

            # 1
            # for i in range(1):
            #     dis = []
            #     verts = set()
            #     faces = set()
            #     for e in bm.edges:
            #         lf = e.link_faces
            #         fa0 = lf[0].calc_area()
            #         fa1 = lf[1].calc_area()
            #         if e.calc_face_angle() * (fa0 + fa1) < 0.1:
            #             # (2 ** i) * 0.1:
            #             if (
            #                 len(lf[0].verts) < 4
            #                 and len(lf[1].verts) < 4
            #                 and e.verts[0].index not in verts
            #                 and e.verts[1].index not in verts
            #                 and lf[0].index not in faces
            #                 and lf[1].index not in faces
            #             ):
            #                 dis.append(e)
            #                 verts.add(e.verts[0].index)
            #                 verts.add(e.verts[1].index)
            #                 faces.add(lf[0].index)
            #                 faces.add(lf[1].index)

            #     bmesh.ops.dissolve_edges(bm, edges=dis, use_verts=True)

            # 2
            # for i in range(1):
            #     # dis = []
            #     verts = set()
            #     # faces = set()
            #     edges = []
            #     for e in bm.edges:
            #         lf = e.link_faces
            #         if len(lf) != 2:
            #             continue

            #         fa0 = lf[0].calc_area()
            #         fa1 = lf[1].calc_area()
            #         if fa0 == 0 or fa1 == 0:
            #             continue

            #         ratio = fa0 / fa1
            #         if ratio > 1.0:
            #             ratio = 1.0 / ratio
            #         if (
            #             e.calc_face_angle() * (fa0 + fa1) * ratio < self.thres
            #             and e.verts[0].index not in verts
            #             and e.verts[1].index not in verts
            #             # and lf[0].index not in faces
            #             # and lf[1].index not in faces
            #         ):
            #             # verts.add(e.verts[0].index)
            #             # verts.add(e.verts[1].index)
            #             # connected = set()
            #             neighbourhood = set(abm.vert_vert(e.verts[0])) |
            # set(abm.vert_vert(e.verts[1]))

            #             for v in list(neighbourhood):
            #                 verts.add(v.index)

            #             # for v in lf[0].verts:
            #             #     verts.add(v.index)

            #             # for v in lf[1].verts:
            #             #     verts.add(v.index)
            #             # faces.add(lf[0].index)
            #             # faces.add(lf[1].index)
            #             edges.append(e)

            #     # bmesh.ops.pointmerge(bm, verts=p, merge_co=(p[0].co + p[1].co) / 2)
            #     bmesh.ops.collapse(bm, edges=edges)

            #     dis = []
            #     for f in bm.faces:
            #         total = 0
            #         for e in f.edges:
            #             if e.is_manifold:
            #                 total += 1
            #         if total < 2:
            #             dis.append(f)

            #     bmesh.ops.delete(bm, geom=dis, context="FACES")
            # bpy.ops.object.mode_set(mode="OBJECT")
            ob = context.object
            vg = None
            if "curve" not in ob.vertex_groups:
                vg = ob.vertex_groups.new(name="curve")
            else:
                vg = ob.vertex_groups["curve"]

            mesh = ob.data

            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)
            norms = afm.read_norms(mesh)

            curve = afm.calc_curvature(verts, edges, norms) - 0.5
            curve = afm.mesh_smooth_filter_variable(curve, verts, edges, 20)
            curve = np.abs(curve)

            # curve -= np.min(curve)
            curve /= np.max(curve)
            curve **= 2.0
            # curve *= 8.0 * self.power

            indices = [v.index for v in bm.verts]

            bpy.ops.object.mode_set(mode="OBJECT")

            for v in indices:
                vg.add([v], curve[v], "REPLACE")

            m_decimate = ob.modifiers.new(name="Decimate", type="DECIMATE")
            m_decimate.ratio = self.thres
            m_decimate.vertex_group = vg.name
            m_decimate.vertex_group_factor = 10.0 * (self.factor ** 4.0)
            m_decimate.invert_vertex_group = True

            # bpy.ops.object.modifier_apply(modifier=m_decimate.name)

            bpy.ops.object.mode_set(mode="EDIT")

        self.payload = _pl


class SelectHidden_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["offset"] = bpy.props.FloatProperty(
            name="Offset", default=0.005, min=0.0, max=1.0
        )

        self.prefix = "select_hidden"
        self.info = "Select hidden"

        self.category = "Vertex data"

        def _pl(self, bm, context):
            raycast.init_with_bm(bm)

            for f in bm.faces:
                f.select = True

            # try to ray cast into the open world
            for f in bm.faces:
                loc = f.calc_center_median() + self.offset * f.normal
                if not all(r[0] for r in raycast.simple_sample(loc, f.normal)):
                    f.select = False

            # ray cast to find any unselected faces
            for f in bm.faces:
                if f.select == False:
                    continue

                loc = f.calc_center_median() + self.offset * f.normal
                if any(
                    bm.faces[i[2]].select == False
                    for i in raycast.simple_sample(loc, f.normal)
                    if i[0] is not None
                ):
                    f.select = False

        self.payload = _pl


class RandomToVCOL_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["blur"] = bpy.props.IntProperty(name="Blur", default=1, min=0, max=10)

        self.prefix = "random_to_vcol"
        self.info = "Random to vertex colors"
        self.fastmesh = True
        self.category = "Vertex data"

        def _pl(self, mesh, context):
            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)

            curve = np.random.random((len(mesh.vertices), 4))
            curve[:, 3] = 1.0

            curve[:, 0] = afm.mesh_smooth_filter_variable(curve[:, 0], verts, edges, self.blur)
            curve[:, 1] = afm.mesh_smooth_filter_variable(curve[:, 1], verts, edges, self.blur)
            curve[:, 2] = afm.mesh_smooth_filter_variable(curve[:, 2], verts, edges, self.blur)

            vcol.write_colors("Random", curve, mesh)

        self.payload = _pl


class CurvatureToVCOL_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["blur"] = bpy.props.IntProperty(name="Blur", default=1, min=0, max=10)
        self.props["alt"] = bpy.props.BoolProperty(name="Alt", default=False)

        self.prefix = "curvature_to_vcol"
        self.info = "Curvature to vertex colors"
        self.category = "Vertex data"
        self.fastmesh = True

        def _pl(self, mesh, context):
            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)
            norms = afm.read_norms(mesh)

            curve = afm.calc_curvature(verts, edges, norms) - 0.5
            curve = afm.mesh_smooth_filter_variable(curve, verts, edges, self.blur)

            if self.alt:
                curve = np.abs(curve)

            curve -= np.min(curve)
            curve /= np.max(curve)

            c = np.ones((len(mesh.vertices), 4))
            c = (c.T * curve.T).T
            vcol.write_colors("Curvature", c, mesh)

        self.payload = _pl


class ThicknessToVCOL_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["blur"] = bpy.props.IntProperty(name="Blur", default=1, min=0, max=10)
        self.props["offset"] = bpy.props.FloatProperty(
            name="Offset", default=0.01, min=0.0, max=1.0
        )

        self.prefix = "thickness_to_vcol"
        self.info = "Thickness to vertex colors"
        self.category = "Vertex data"
        self.fastmesh = True

        def _pl(self, mesh, context):
            thick = np.zeros(len(mesh.vertices), dtype=np.float64)
            with au.Mode_set("EDIT"):
                with abm.Bmesh_from_edit(mesh) as bm:
                    raycast.init_with_bm(bm)

                    for f in bm.faces:
                        loc = f.calc_center_median() - self.offset * f.normal
                        res = raycast.simple_sample(loc, -f.normal)

                        # total distance
                        total = sum(i[3] for i in res if i[0] is not None)

                        for v in f.verts:
                            thick[v.index] += total

            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)
            thick = afm.mesh_smooth_filter_variable(thick, verts, edges, self.blur)

            thick /= np.max(thick)

            c = np.ones((len(mesh.vertices), 4))
            c = (c.T * thick.T).T
            vcol.write_colors("Thickness", c, mesh)

        self.payload = _pl


class AmbientOcclusionToVCOL_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.props["blur"] = bpy.props.IntProperty(name="Blur", default=1, min=0, max=10)
        self.props["offset"] = bpy.props.FloatProperty(
            name="Offset", default=0.01, min=0.0, max=1.0
        )
        self.props["dist"] = bpy.props.FloatProperty(name="Distance", default=1.0, min=0.0)

        self.prefix = "ao_to_vcol"
        self.info = "Ambient occlusion to vertex colors"
        self.category = "Vertex data"
        self.fastmesh = True

        def _pl(self, mesh, context):
            ao = np.zeros(len(mesh.vertices), dtype=np.float64)
            with au.Mode_set("EDIT"):
                with abm.Bmesh_from_edit(mesh) as bm:
                    raycast.init_with_bm(bm)

                    for f in bm.faces:
                        loc = f.calc_center_median() + self.offset * f.normal
                        res = raycast.simple_sample(loc, f.normal)

                        # total distance
                        total = sum(
                            i[3] / self.dist if i[0] is not None and i[3] < self.dist else 1.0
                            for i in res
                        )

                        for v in f.verts:
                            ao[v.index] += total

            verts = afm.read_verts(mesh)
            edges = afm.read_edges(mesh)
            ao = afm.mesh_smooth_filter_variable(ao, verts, edges, self.blur)

            ao -= np.min(ao)
            ao /= np.max(ao)

            c = np.ones((len(mesh.vertices), 4))
            c = (c.T * ao.T).T
            vcol.write_colors("AO", c, mesh)

        self.payload = _pl


# Detect all relevant classes in namespace
load_these = []
for name, obj in locals().copy().items():
    if hasattr(obj, "__bases__") and obj.__bases__[0].__name__ == "MeshOperatorGenerator":
        load_these.append(obj)

pbuild = PanelBuilder("mesh_refine_toolbox", load_these)
OBUILD_PT_MeshRefineToolbox = pbuild.create_panel()


def register():
    pbuild.register_params()
    bpy.utils.register_class(OBUILD_PT_MeshRefineToolbox)


def unregister():
    pbuild.unregister_params()
    bpy.utils.unregister_class(OBUILD_PT_MeshRefineToolbox)
