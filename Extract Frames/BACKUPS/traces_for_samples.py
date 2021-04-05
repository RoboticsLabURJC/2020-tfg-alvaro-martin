''' TEST NEW traces for sample'''
'''
        os.chdir(real_path)
        cv2.circle(black_img, (real_y, real_x), 1, (246, 209, 81), -1)
        cv2.imwrite('Real_Trails' + str(i) + '.png', black_img)
        os.chdir(predict_path)
        cv2.circle(black_img_2, (pr_y, pr_x), 1, (15, 232, 253), -1)
        cv2.imwrite('Predicted_Trails' + str(i) + '.png', black_img_2)

'''

import os
import numpy as np
import cv2

folder_path = '/Users/Martin/Desktop/Generator_10/Frames_dataset/linear_point_255_fix_1000_80_120_30GAP/linear_30_[None]_test/modeled_samples/'

"""Create new image filled with certain color in RGB"""
def create_blank(width, height, rgb_color=(0, 0, 0)):

    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

''' Resize image '''
def resize(width, height):

    dim = (width, height)
    resized = cv2.resize(black_image, dim, interpolation = cv2.INTER_AREA)

    return resized

if __name__ == '__main__':

    i = 0

    for sample in os.listdir(folder_path):

        if sample.endswith(".txt"):

            real_path = folder_path + '/_real_trails'
            os.makedirs(real_path,exist_ok=True)
            predict_path = folder_path + '/_predicted_trails'
            os.makedirs(predict_path,exist_ok=True)
            os.chdir(folder_path)

            i += 1
            print(sample + str(i))

            w, h = 120, 80
            black = (0, 0, 0)
            black_img = np.zeros((h, w, 3), np.uint8)
            black_img_2 = np.zeros((h, w, 3), np.uint8)

            # Since OpenCV uses BGR, convert the color first
            color = tuple(reversed(black))
            # Fill image with color
            black_img[:] = color
            #black_img_2[:] = color

            with open(sample) as fname:
            	lineas = fname.readlines()
            	for linea in lineas:
                    x = str(linea.split(',')[0])
                    y = str(linea.split(',')[1])
                    if x != 'x' and y != 'y':
                        x = int(x)
                        y = int(y)

                        real_path = folder_path + '/_real_trails'
                        os.makedirs(real_path,exist_ok=True)
                        predict_path = folder_path + '/_predicted_trails'
                        os.makedirs(predict_path,exist_ok=True)

                        os.chdir(real_path)
                        cv2.circle(black_img, (x, y), 1, (246, 209, 81), -1)
                        cv2.imwrite('Real_Trails' + str(i) + '.png', black_img)
                        #os.chdir(predict_path)
                        #cv2.circle(black_img_2, (pr_y, pr_x), 1, (15, 232, 253), -1)
                        #cv2.imwrite('Predicted_Trails' + str(i) + '.png', black_img_2)
