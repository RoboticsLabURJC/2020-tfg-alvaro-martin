import cv2

img = cv2.imread("MVI_2502.MP4_frames/video_frames_GREY 84.png", 0)
#print(img) #since the image is grayscale, we need only one channel and the value '0' indicates just that
for i in range (img.shape[0]): #traverses through height of the image
    for j in range (img.shape[1]): #traverses through width of the image
        print (img[i][j])
