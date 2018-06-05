import cv2

# 0 - for black and white
#1 - for color pic
img = cv2.imread("galaxy.jpg", 0)
print (type(img))
print (img)
print (img.shape)
print (img.ndim)

# show image on the screen
cv2.imshow("Galaxy", img)
cv2.waitKey(0)  # image displays and user can press any button e.g 'B' to exit
#cv2.waitKey(2000) # image shown for 2000 milisecond
cv2.destroyAllWindows()

# show resized image
resized_image = cv2.resize(img, (1000,500))
cv2.imshow("Galaxy", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# show resized image: resolution by /2
resized_image = cv2.resize(img, (int(img.shape[0]/2),int(img.shape[1]/2)))
cv2.imshow("Galaxy", resized_image)
cv2.imwrite("galaxy_resized.jpg", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
