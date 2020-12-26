from urllib import request
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_IMAGE():

    img = cv.imread('bilder/20201208_141922.jpg', 1)

    # img = img[33:412, 93:599]

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #
    # ret0, thresh0 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    #
    # kernel = np.ones((2, 2), np.uint8)
    # erosion = cv.erode(thresh0, kernel, iterations=1)
    # dilation = cv.dilate(thresh0, kernel, iterations=1)
    #
    # erg = cv.dilate(erosion, kernel, iterations=1)

    # print(kernel)

    # cv2.line(img, (0, 0), (150, 150), (0, 255, 0), 10) #Linie
    # cv2.rectangle(img, (15, 25), (200, 150), (0, 250, 0), 5)   #Rechteck
    # cv2.circle(img, (100, 63), 55, (0, 255, 0), -1)    #Kreis
    # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)  #P_gon
    # cv2.polylines(img, [pts], True, (0, 255, 0), 1)    # True Parameter (start-stop) verbinden
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, 'TEST!', (0, 130), font, 1, (0, 255, 0))

    # titel = ['Original', 'Gray', 'Binary', 'Erosion', 'Dilation', 'ERG']
    # bilder = [img, gray, thresh0, erosion, dilation, erg]
    #
    # for k, i in enumerate(bilder):
    #    plt.subplot(2, 3, k+1)
    #    plt.imshow(i, 'gray')
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.title(titel[k])

    #cv.imshow('Bild', img)
    #cv.imwrite('neu.png', img)
    return img

def detectDartboard(IM):

    IM_blur = cv.medianBlur(IM, 5)
    # convert to HSV
    base_frame_hsv = cv.cvtColor(IM_blur, cv.COLOR_BGR2HSV)
    # Extract Green
    green_thres_low = int(60 / 255. * 180)
    green_thres_high = int(180 / 255. * 180)
    green_min = np.array([green_thres_low, 100, 100], np.uint8)
    green_max = np.array([green_thres_high, 255, 255], np.uint8)
    frame_threshed_green = cv.inRange(base_frame_hsv, green_min, green_max)
    kernel = np.ones((2, 2), np.uint8)
    erosion_green = cv.erode(frame_threshed_green, kernel, iterations=1)
    frame_threshed_green_new = cv.dilate(erosion_green, kernel, iterations=1)
    # Extract Red
    red_thres_low = int(0 / 255. * 180)
    red_thres_high = int(20 / 255. * 180)
    red_min = np.array([red_thres_low, 100, 100], np.uint8)
    red_max = np.array([red_thres_high, 255, 255], np.uint8)
    frame_threshed_red = cv.inRange(base_frame_hsv, red_min, red_max)
    kernel = np.ones((2, 2), np.uint8)
    erosion_red = cv.erode(frame_threshed_red, kernel, iterations=1)
    frame_threshed_red_new = cv.dilate(erosion_red, kernel, iterations=1)
    # Combine
    combined = frame_threshed_red_new + frame_threshed_green_new
    #combined = frame_threshed_red + frame_threshed_green
    # Close
    kernel = np.ones((100, 100), np.uint8)
    closing = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel)
    # find contours
    ret, thresh = cv.threshold(combined,
                               127,
                               255,
                               0)
    contours, hierarchy = cv.findContours(closing.copy(),
                                          cv.RETR_LIST,
                                          cv.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        print(len(contours[i]))
        if 500 < len(contours[i]) < 10000:
            final = cv.drawContours(IM, contours[i], -1, (0, 255, 0), 20)
        else:
            pass

            # max_cont = -1
             # max_idx = 0
             # for i in range(len(contours)):
             #     length = cv.arcLength(contours[i], True)
             #     if  length > max_cont:
             #         max_idx = i
             #         max_cont = length
             # x,y,w,h = cv.boundingRect(contours[max_idx])
             # x = x-Dartboard_Detector.ENV['DETECTION_OFFSET']
             # y = y-Dartboard_Detector.ENV['DETECTION_OFFSET']
             # w = w+int(2*Dartboard_Detector.ENV['DETECTION_OFFSET'])
             # h = h+int(2*Dartboard_Detector.ENV['DETECTION_OFFSET'])
             # return x,y,w,h,closing,frame_threshed_green,frame_threshed_red

    #cv.imshow('Median Blur', IM_blur)
    #cv.imshow('Frame Green', frame_threshed_green)
    #cv.imshow('Frame Green NEW', frame_threshed_green_new)
    #cv.imshow('Frame Red', frame_threshed_red)
    #cv.imshow('Frame Red NEW', frame_threshed_red_new)
    #cv.imshow('Combine', combined)
    #cv.imshow('Closing', closing)
    cv.imshow('Final', final)

if __name__ == '__main__':

    detectDartboard(get_IMAGE())

    cv.waitKey(0)
    cv.destroyAllWindows()
