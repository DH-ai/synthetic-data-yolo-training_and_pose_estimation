import cv2
import numpy as np

# ---------------------------
# 1. Camera calibration data (replace with your own!)
# ---------------------------

K = np.array([[640, 0, 320],
              [0, 480, 240],
              [0, 0, 1]], dtype=np.float32)
dist = np.zeros((4,1))  # no distortion as example

# ---------------------------
# 2. ArUco setup
# ---------------------------
ARUCO_DICT = cv2.aruco.DICT_6X6_250
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Real marker side length (in same units as translation results, e.g. cm)
MARKER_LEN = 5.0   # 5 cm

# ---------------------------
# 3. Function to draw pose axes
# ---------------------------
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

# ---------------------------
# 4. Main loop (using webcam)
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        # Estimate pose for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LEN, K, dist)
        
        for i in range(len(ids)):
            # Draw detected marker outline
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Draw 3D axis
            frame = draw_axis(frame, rvecs[i], tvecs[i], K, dist, length=MARKER_LEN*0.8)
            
            # Print rotation vector (Rodrigues) and translation vector
            print(f"ID {ids[i][0]}:")
            print(f"  rvec = {rvecs[i].flatten()}")
            print(f"  tvec = {tvecs[i].flatten()} (units = marker length units)")
    
    cv2.imshow("ArUco Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()