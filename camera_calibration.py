import cv2
import numpy as np
import matplotlib.image as mpimg


class CameraCalibration:
    def __init__(self):
        self.mtx = None
        self.dist = None

    def cal_undistort(self, image_paths, n_rows: int, n_cols: int):
        # takes an image, object points, and image points
        # performs the camera calibration, image distortion correction and
        # returns the undistorted image
        objpoints = []
        imgpoints = []

        objp = _object_points(n_rows, n_cols)

        for image_path in image_paths:
            image = mpimg.imread(image_path)
            image_shape = _grayscale_shape(image)
            ret, corners = _find_chessboard_corners(image, n_cols, n_rows)

            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

        self.dist = dist
        self.mtx = mtx

    def transform(self, image):
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return undist


def _object_points(n_rows, n_cols):
    objp = np.zeros((n_rows * n_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    return objp


def _find_chessboard_corners(img, n_cols, n_rows):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (n_cols, n_rows), None)
    return ret, corners


def _grayscale_shape(img):
    # create a 2-tuple as shape of gray scale image
    return img.shape[1::-1]
