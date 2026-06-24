import trimesh
import numpy as np
import os 

objs =[
    "heart",
    "semicircle",
    "triangle",
]

for obj in objs:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"/output/bop/models/{obj}.ply")


    path = os.path.dirname(os.path.abspath(__file__)) + f"/output/bop/models/{obj}.ply"
    # print(path)
    if os.path.exists(path) is False:
        print(f"Mesh file does not exist: {path}")
        continue
    # print(f"Loading mesh from: {path}")
    with open(path, 'rb') as f:
        mesh = trimesh.load(f, file_type='ply')
    # mesh = trimesh.load(path)

    verts = mesh.vertices

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)

    size = maxs - mins

    print(f"Object: {obj}")
    print(mesh.bounds)
    print("min_x", mins[0])
    print("min_y", mins[1])
    print("min_z", mins[2])

    print("size_x", size[0])
    print("size_y", size[1])
    print("size_z", size[2])

    diameter = np.max(
        np.linalg.norm(
            verts[:, None, :] - verts[None, :, :],
            axis=2
        )
    )

    print("diameter", diameter)
