import cv2
import matplotlib.pyplot as plt

img = cv2.imread('examples/center.jpg')

#cv2.cvtColor(cv2.imread('examples/center.jpg'), cv2.COLOR_BGR2RGB)
#cv2.imwrite('examples/flipped.jpg', cv2.flip(img,1))

#img = crop_img = img[60:140, :]

#cv2.imwrite('examples/cropped.jpg', img)

plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2HLS) )
plt.axis('off')
plt.show()
#flip_img_center = cv2.flip(img_center,1)
