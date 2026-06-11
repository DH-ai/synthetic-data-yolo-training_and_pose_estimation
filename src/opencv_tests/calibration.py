import cv2
import numpy as np
import os 

import glob 
# Prepare your calibration images
chessboard_size = (9, 6)
calibration_images = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', f) for f in ['image1.png', 'image1.png', 'image1.png']]  # Your images
# print(calibration_images)
# exit()

# Arrays to store object points and image points
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