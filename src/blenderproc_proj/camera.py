import numpy as np
import blenderproc as bproc 

K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
        [0.0, 2482.3917472975795, 629.72289542481894],
        [0.0, 0.0, 1.0]],dtype=np.float64)

dist =   np.array([[ -0.091539129459748417, 1.6518788910916924,
    -0.00096826424151305102, -0.0023115236516727399,
    -7.1086932137755738]]) 
WIDTH = 1920
HEIGHT = 1200


def set_camera():
    """Set BlenderProc camera intrinsics from the calibrated K matrix."""
    bproc.camera.set_intrinsics_from_K_matrix(K, WIDTH, HEIGHT)



# main objective is to make a camera implmentation sub file where im launching the camera and givign it pose retriveing tvec rvec etc, and making code more managembale 


# transformation code:
# Tco = object_to_camera(Tcw, Two)
# Tcw = world_to_camera(...)
# Twc = camera_to_world(Tcw)
# Two = world_to_object(...)
# Toc = camera_to_object(Tcw, Tco)




# def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
#     """Build a 4x4 homogeneous transform from rotation and translation."""
#     T = np.eye(4, dtype=np.float64)
#     T[:3, :3] = np.asarray(R, dtype=np.float64)
#     T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
#     return T


# def invert_transform(T: np.ndarray) -> np.ndarray:
#     """Invert a rigid 4x4 homogeneous transform."""
#     T = np.asarray(T, dtype=np.float64)
#     R = T[:3, :3]
#     t = T[:3, 3]
#     T_inv = np.eye(4, dtype=np.float64)
#     T_inv[:3, :3] = R.T
#     T_inv[:3, 3] = -R.T @ t
#     return T_inv


# def world_to_camera(T_cw: np.ndarray) -> np.ndarray:
#     """World -> camera (Tcw)."""
#     return np.asarray(T_cw, dtype=np.float64)


# def camera_to_world(T_cw: np.ndarray) -> np.ndarray:
#     """Camera -> world (Twc)."""
#     return invert_transform(T_cw)

# def world_to_object(T_wo: np.ndarray) -> np.ndarray:
#     """World -> object (Two)."""
#     return np.asarray(T_wo, dtype=np.float64)


# def object_to_world(T_wo: np.ndarray) -> np.ndarray:
#     """Object -> world (Tow)."""
#     return invert_transform(T_wo)


# def object_to_camera(T_cw: np.ndarray, T_wo: np.ndarray) -> np.ndarray:
#     """Object -> camera (Tco) from world->camera and world->object."""
#     return world_to_camera(T_cw) @ object_to_world(T_wo)


# def camera_to_object(T_cw: np.ndarray, T_co: np.ndarray) -> np.ndarray:
#     """Camera -> object (Toc)."""
#     return invert_transform(T_co)


# def rvec_tvec_from_transform(T: np.ndarray):
#     """Return OpenCV-style rvec/tvec from a 4x4 transform."""
#     import cv2

#     T = np.asarray(T, dtype=np.float64)
#     rvec, _ = cv2.Rodrigues(T[:3, :3])
#     tvec = T[:3, 3].reshape(3, 1)
#     return rvec, tvec


# def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
#     """Build a 4x4 transform from OpenCV-style rvec/tvec."""
#     import cv2

#     R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
#     return make_transform(R, np.asarray(tvec, dtype=np.float64).reshape(3))


# Tow = object_to_world(Two)