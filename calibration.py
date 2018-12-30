import numpy as np
import cv2 as cv
import glob
import pickle
import json
import time

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
print images
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    print ret, corners
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
calib_output = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
ret, mtx, dist, rvecs, tvecs = calib_output

with open('calib.pickle','w') as picklefile:
    pickle.dump(calib_output, picklefile)

for (name, v) in zip(['ret', 'mtx', 'dist', 'rvecs', 'tvecs'], calib_output):
    print 'to json: {name}'.format(**locals())
    t = type(v)
    with open('calib_{name}.json'.format(**locals()),'w') as jsonfile:
        try:
            if t is np.ndarray:
                json.dump(v.tolist(), jsonfile)
            elif t is list:
                json.dump(map(lambda x: x.tolist(), v), jsonfile)
            else:
                json.dump(v, jsonfile)
        except Exception as exn:
            print "ERROR: ", name, v
            raise exn

#img = cv.imread('2018-11-25-130553.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
#print dist
#print roi
cv.imwrite('calibresult.png', dst)

#with np.load('calib_mtx') as X:
#    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    

    
