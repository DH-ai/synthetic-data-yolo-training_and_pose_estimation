import open3d as o3d, os.path as osp



mdir = "src/output/bop/models"


for i in (1, 2, 3):
    p = osp.join(mdir, f"obj_{i:06d}.ply")
    m = o3d.io.read_triangle_mesh(p)
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(p, m, write_vertex_normals=True)