import cv2 as cv
import numpy as np
import test_einlesen

def detectDartboard(IM):
    #IM gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
        #print(len(contours[i]))
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
    #cv.imshow('Final', final)
    return final, combined, thresh, closing, frame_threshed_red_new, frame_threshed_green_new

if __name__ == '__main__':

    final, combined, thresh, closing, frame_threshed_red_new, frame_threshed_green_new = detectDartboard(test_einlesen.get_IMAGE('bilder/20201208_141922.jpg', -1))
    title = ['final', 'combined', 'thresh', 'closing', 'frame_threshed_red_new', 'frame_threshed_green_new']
    #bilder = [combined, thresh, closing, frame_threshed_red_new, frame_threshed_green_new]
    bilder = [final, combined]

    for k, i in enumerate(bilder):
        cv.namedWindow(title[k], cv.WINDOW_NORMAL)
        cv.moveWindow(title[k], 20, 20)
        cv.resizeWindow(title[k], 600, 600)
        cv.imshow(title[k], i)

    cv.waitKey(0)
    cv.destroyAllWindows()