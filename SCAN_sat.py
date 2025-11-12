import os, cv2
import numpy as np
path = r"C:\Users\Simon\Pictures\SCANsat_Images"

file_list = []

for filename in os.listdir(path):
    file_list.append(filename)

file_list[1]
img = cv2.imread(os.path.join(path, file_list[3]))
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 150
params.maxArea = 3000
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByConvexity = True
params.minConvexity = 0.37
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()