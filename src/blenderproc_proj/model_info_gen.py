import trimesh
import numpy as np
import os 
import json 


objs =[
    "obj_000001",
    "obj_000002",
    "obj_000003",
]


class ModelInfo:
    def __init__(self, diameter, min_x, min_y, min_z, size_x, size_y, size_z):
        self.diameter = diameter
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
model_dict  = {}

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

    model_info = ModelInfo(diameter, mins[0], mins[1], mins[2], size[0], size[1], size[2])
    model_dict[objs.index(obj) + 1] = model_info.__dict__


# print (model_dict)
print("writing model info to json file")
with open(os.path.dirname(os.path.abspath(__file__)) + "/output/bop/models/models_info.json", 'w') as f:
    # clean the file before writing
    f.truncate(0)
    
    json.dump(model_dict, f, indent=4)