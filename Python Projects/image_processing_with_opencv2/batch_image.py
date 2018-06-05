
import cv2
import glob

images = glob.glob("*.jpg")
for img in images:
    im = cv2.imread(img,0)
    re = cv2.resize(im, (100, 100))
    cv2.imshow("Hey", re)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.imwrite("resize_"+img, re)
