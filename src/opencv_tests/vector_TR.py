import os
import cv2
import numpy as np


# K and dist: from mech eye sdk
K = np.array([[2430.38, 0.0, 969.89],
            [0.0, 2431.72, 619.58],
            [0.0, 0.0, 1.0]],dtype=np.float64)
dist = np.zeros(5, dtype=np.float64)   # distortion coeffs

# ArUco setup
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015



aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), 
                               SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

# Real marker side length (in same units as translation results, e.g. cm)
MARKER_LEN = 5.0   # 5 cm



# Function to draw pose axes
def draw_axis(img, rvec, tvec, K, dist, length=3.0):
    """
    Draw 3D axes (X: red, Y: green, Z: blue) on the image.
    length: axis length in the same units as tvec (e.g., cm)
    """
    # Define 3D points in marker coordinate system (origin at marker centre)
    axis_points = np.float32([[length,0,0], [0,length,0], [0,0,-length]]).reshape(-1,3)
    # Project to 2D image points
    img_pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1,2).astype(int)
    # Origin (centre of marker)
    origin, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, K, dist)
    origin = tuple(origin.reshape(2).astype(int))
    
    # Draw lines
    cv2.line(img, origin, tuple(img_pts[0]), (0,0,255), 3)  # X red
    cv2.line(img, origin, tuple(img_pts[1]), (0,255,0), 3)  # Y green
    cv2.line(img, origin, tuple(img_pts[2]), (255,0,0), 3)  # Z blue
    return img


path = os.path.join(os.path.dirname(__file__), "media/charuco_pose.png")
print(f"Loading test image from: {path}")
img = cv2.imread(path)  # Load your test image here

if img is None:
    raise FileNotFoundError("Could not load image, check path and filename")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect ArUco markers
# corners, ids, rejected = detector.detectMarkers(img) # for regular ArUco


charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 3:
    # Estimate pose
    # `cv2.aruco.estimatePoseCharucoBoard` is still the function used internally.
    # Deprecated in  modern API
    # ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
    #     charuco_corners, charuco_ids, board, K, dist, None, None)
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)

    

    
    if success:
        # img = draw_axis(img, rvec, tvec, K, dist)
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.1)
        print(f"ChArUco rvec={rvec.flatten()}, tvec={tvec.flatten()}")



# if ids is not None:

#     # Estimate pose for each marker
    # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LEN, K, dist)
    
#     for i in range(len(ids)):
#         # Draw detected marker outline
#         cv2.aruco.drawDetectedMarkers(img, corners, ids)
        
#         # Draw 3D axis
#         img = draw_axis(img, rvecs[i], tvecs[i], K, dist, length=MARKER_LEN*0.8)
        
#         # Print rotation vector (Rodrigues) and translation vector
#         print(f"ID {ids[i][0]}:")
#         print(f"  rvec = {rvecs[i].flatten()}")
#         print(f"  tvec = {tvecs[i].flatten()} (units = marker length units)")

cv2.imshow("ArUco Pose Estimation", img)
cv2.waitKey(0)


cv2.destroyAllWindows()