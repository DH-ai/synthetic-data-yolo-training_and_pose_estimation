import open3d as o3d

mesh = o3d.io.read_triangle_mesh("object.stl")
pcd_model = mesh.sample_points_uniformly(number_of_points=10000)