import os
import cv2
import numpy as np


checkboard = False
mecheye = False # brute method of chaing K and dist 
if checkboard:

    ## Testing K and dist from the checkboard test 
    K = np.array([[2.87094566e+03, 0.00000000e+00, 1.04248232e+03],
                [0.00000000e+00, 2.88403367e+03, 7.04362990e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
    dist =   np.array([[ 2.03276531e-01, -3.47445013e+00,  6.88249751e-03,  1.74410119e-02,
    3.35986406e+01]])    
elif mecheye:
    # K and dist: from mech eye sdk getting transformation vector at around 99.7 cm witch 2cm error, which is pretty good for our use case.
    K = np.array([[2430.38, 0.0, 969.89],
            [0.0, 2431.72, 619.58],
            [0.0, 0.0, 1.0]],dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)   # distortion coeffs
else: # values for charuca calibration
    K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
            [0.0, 2482.3917472975795, 629.72289542481894],
            [0.0, 0.0, 1.0]],dtype=np.float64)

    dist =   np.array([[ -0.091539129459748417, 1.6518788910916924,
       -0.00096826424151305102, -0.0023115236516727399,
       -7.1086932137755738]]) 

# ArUco setup
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.022
MARKER_LENGTH = 0.011



aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), 
                               SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

# Real marker side length (in same units as translation results, e.g. cm)
# MARKER_LEN = 50.0   # 5 cm


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


def process_frame(img,draw_rectangle:bool=False):
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 3:
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)

        if success:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.1)
            
            print(f"ChArUco rvec={rvec.flatten()}, tvec={tvec.flatten()}")
        if draw_rectangle:
            # Draw a rectangle around the detected board
            # print(charuco_corners,marker_corners)
            img = cv2.polylines(img, [charuco_corners.astype(int)], isClosed=True, color=(255,0,255), thickness=2)
            # print((tuple(marker_corners[0][0][0].astype(float))))
            # print(marker_corners[0][0][0].astype(float))
            for marker_corner in marker_corners:
                pt1 = tuple(marker_corner[0][0].astype(int))   # (x1, y1)
                pt2 = tuple(marker_corner[0][2].astype(int))   # (x2, y2)
                cv2.rectangle(img, pt1, pt2, (0, 255, 255), 2)

    return img, rvec, tvec


def run_image(image_path=None):
    if image_path is None:
        image_path = os.path.join(os.path.dirname(__file__), "media_charucoBoard/rgb_image_20260611_200754_240.png")

    print(f"Loading test image from: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Could not load image, check path and filename")

    img, rvec, tvec = process_frame(img,True)
    distance = np.linalg.norm(tvec)   # in same units as SQUARE_LENGTH
    cv2.putText(img, f"Distance to board: {distance:.3f} meters", (60,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    print(f"{rvec}, {tvec}")
    cv2.line(img, (600,), (200, 50), (255, 0, 0), 2)  # Blue line for X-axis
    cv2.imshow("ArUco Pose Estimation", img)
    
    

    
    while True:
    # Wait indefinitely for a key press
        key = cv2.waitKey(0) & 0xFF
        
        # Close window ONLY if 'q' (or 'Q') is pressed
        if key == ord('q') or key == ord('Q'):
            break
    cv2.destroyAllWindows()


def run_webcam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, rvec, tvec = process_frame(frame)
            cv2.imshow("ArUco Pose Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def calibration():
    # Add functionality to test with webcam
    pass
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChArUco pose estimation demo")
    parser.add_argument("--webcam", action="store_true", help="Use live webcam feed")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    if args.webcam:
        run_webcam(args.camera_index)
    else:
        run_image(args.image)