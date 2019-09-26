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


class DelaunayCriterion_OP(mesh_ops.MeshOperatorGenerator):
    def generate(self):
        self.prefix = "delaunay_criterion"
        self.info = "Flip edges that don't meet the criterion"

        self.category = "Cleanup"

        def _pl(self, bm, context):
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            flips = 0
            for e in bm.edges:
                f = e.link_faces
                if len(f) == 2 and len(f[0].verts) == 3 and len(f[1].verts) == 3:
                    l0 = [i for i in f[0].loops if i.vert != e.verts[0] and i.vert != e.verts[1]][0]
                    l1 = [i for i in f[1].loops if i.vert != e.verts[0] and i.vert != e.verts[1]][0]

                    a0 = l0.calc_angle()
                    a1 = l1.calc_angle()
                    if a0 + a1 > np.pi:
                        bmesh.utils.edge_rotate(e, True)
                        flips += 1
            print("flips:", flips)

        self.payload = _pl