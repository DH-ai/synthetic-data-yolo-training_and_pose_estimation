import trimesh
import numpy as np
import os 

objs =[
    heart_shape.on
]

for obj in objs:
    mesh = trimesh.load(f"/assets/models/{obj}.ply")

    verts = mesh.vertices

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)

    size = maxs - mins

    print(f"Object: {obj}")
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
