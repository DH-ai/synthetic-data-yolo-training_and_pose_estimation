import cv2
import numpy as np
import os 

import glob 
# Prepare your calibration images
chessboard_size = (9, 6) # Number of inner corners per chessboard row and column
calibration_images = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media_chessBoard', f) for f in ['image1.png', 'image1.png', 'image1.png']]  # Your images




# ================= Configuration =================

SQUARES_X = 7          
SQUARES_Y = 5          
SQUARE_LENGTH = 0.022   
MARKER_LENGTH = 0.011  
ARUCO_DICT = cv2.aruco.DICT_6X6_250


IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media_charucoBoard', '*.png')


# ================= Charuco Calibration Part =================

# Creating charuco board object and detector 
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = cv2.aruco.CharucoDetector(board)


# Arr to store detected corners and thier IDs
all_charuco_corners = []
all_charuco_ids = []
image_size = None

# Here well do the processing
charucoBoard = True
if charucoBoard:

    image_paths = glob.glob(IMAGES_PATH)
    for img_path in image_paths:
        img = cv2.imread(img_path)  
        if img is None:
            continue
        if image_size is None:
            image_size = img.shape[1], img.shape[0]  # (width, height)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting the charuco corners and ids
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)

        if charuco_corners is not None and len(charuco_corners) > 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"Good frame found in {os.path.basename(img_path)} with {len(charuco_corners)} corners.")
        else:
            print(f"Not enough ChArUco corners found in {os.path.basename(img_path)}. Skipping.")



    # performing calibration 

    if len(all_charuco_corners) > 0:
        print(f"\nCalibrating with {len(all_charuco_corners)} valid frames...")

        # Initialize the camera matrix and distortion coefficients
        camera_matrix = np.eye(3, dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # # Run the calibration , works in the older version of open cv not int new
        # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        #     all_charuco_corners, all_charuco_ids, board, image_size, camera_matrix, dist_coeffs
        # )

        objpoints = []
        imgpoints = []

        # Iterate through all detected frames
        for corners, ids in zip(all_charuco_corners, all_charuco_ids):
            # Get object (3D) and image (2D) points for the ChArUco corners
            obj_points, img_points = board.matchImagePoints(corners, ids)
            if obj_points is not None and img_points is not None:
                objpoints.append(obj_points)
                imgpoints.append(img_points)
    
        # Perform calibration using the standard OpenCV function
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )


        if ret:
            print("\nCalibration successful!")
            print(f"Camera Matrix (K):\n{camera_matrix}")
            print(f"Distortion Coefficients (dist):\n{dist_coeffs}")
            print(f"Reprojection Error: {ret:.4f} pixels")
            print(f"Average distance to board: {np.mean([np.linalg.norm(tvec) for tvec in tvecs]):.4f} units")
            print("Rotation Vectors:", rvecs)
            print("Translation Vectors:", tvecs)
            
            # Optional: Save the calibration results to a file
            cv2.fileStorage = cv2.FileStorage("calibration_charuco.yaml", cv2.FILE_STORAGE_WRITE)
            cv2.fileStorage.write("camera_matrix", camera_matrix)
            cv2.fileStorage.write("dist_coeffs", dist_coeffs)
            cv2.fileStorage.release()
            print("Calibration parameters saved to 'calibration_charuco.yaml'")
        else:
            print("Calibration failed. Try using more images with better coverage.")
    else:
        print("No valid frames found for calibration. Check your image path and board visibility.")






else:
    print("Calibration with Chessboard")
    #  Arrays to store object points and image points



    objpoints = []
    imgpoints = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Process each image
    for fname in calibration_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)


        # Testing the finded corners
        if False:

            for corner in corners:
                # print(tuple(corner[0]))
                cv2.circle(img, corner[0].astype(int), 10, (125, 55, 25), thickness=-1)
            
            
            cv2.imshow("Corners", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



    # for fname in calibration_images:
    #     img = cv2.imread(fname)
    #     h, w = img.shape[:2]
    #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    #     print("Optimal new camera matrix:", newcameramtx)
    #     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #     x, y, w, h = roi
    #     dst = dst[y:y+h, x:x+w]
    #     cv2.imshow("Undistorted", dst)
    #     cv2.imshow("Original", img)
    #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # print("Camera calibration successful:", ret)

    print("Camera Matrix:", mtx)
    print("Distortion Coefficients:", dist)
    # print("Rotation Vectors:", rvecs)
    print("Translation Vectors:", tvecs)
    print(f"distance to board: {np.linalg.norm(tvecs[0])} meters")


    ## add functionality to test with webcam