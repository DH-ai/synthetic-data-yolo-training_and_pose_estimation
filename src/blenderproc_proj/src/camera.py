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
    bproc.camera.set_intrinsics_from_K_matrix(K, WIDTH, HEIGHT)

# bproc.camera.



# main objective is to make a camera implmentation sub file where im launching the camera and givign it pose retriveing tvec rvec etc, and making code more managembale 