import cv2 as cv
import time
import numpy as np
import schedule
import serial

def view_image(img, name):
    # flipped = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    # gray = cv.cvtColor(flipped, cv.COLOR_BGR2GRAY)
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    #cv.moveWindow(name, 20, 20)
    cv.resizeWindow(name, 400, 400)
    cv.imshow(name, img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

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
    #cv.imshow('Final', final)
    return combined

def computeDifference(grey1, grey2):

    # blur
    blur = 5
    grey2 = cv.medianBlur(grey2, blur)
    grey1 = cv.medianBlur(grey1, blur)
    # normalize
    grey1 = cv.equalizeHist(grey1)
    grey2 = cv.equalizeHist(grey2)
    clahe = cv.createCLAHE(20, (10, 10))
    # clahe
    grey1 = clahe.apply(grey1)
    grey2 = clahe.apply(grey2)
    # #diff
    diff = cv.subtract(grey2, grey1) + cv.subtract(grey1, grey2)
    ret2, dif_thred = cv.threshold(diff, 75, 255, cv.THRESH_BINARY)

    return dif_thred

def get_IMAGE(name, param):

    img = cv.imread(name, param)

    # img = img[33:412, 93:599]

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

def get_VIDEO(pfad):

    cap = cv.VideoCapture(pfad)

    while (cap.isOpened()):

        ret, frame = cap.read()
        ret1, frame1 = cap.read()
        flipped = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        gray = cv.cvtColor(flipped, cv.COLOR_BGR2GRAY)
        flipped1 = cv.rotate(frame1, cv.ROTATE_90_CLOCKWISE)
        gray1 = cv.cvtColor(flipped1, cv.COLOR_BGR2GRAY)

        im = detectDartboard(flipped)
        #im = computeDifference(gray, gray1)

        name = 'frame'
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.moveWindow(name, 20, 20)
        cv.resizeWindow(name, 600, 600)
        cv.imshow('frame', flipped)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        #time.sleep(0.1)

    cap.release()
    cv.destroyAllWindows()

def get_frame():

    cap = cv.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    return frame

def view_image(im, name):

    #flipped = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    #gray = cv.cvtColor(flipped, cv.COLOR_BGR2GRAY)

    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.moveWindow(name, 20, 20)
    cv.resizeWindow(name, 600, 600)
    cv.imshow(name, im)

def corrected(img1, img2):  # perspektive von bild 2 wird auf bild 1 angepasst. (korrigiert)

    MIN_MATCHES = 50

    orb = cv.ORB_create(nfeatures=500)  # 500features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)

    search_params = {}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # wie gut liegen die features aneinander
            good_matches.append(m)

    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
        corrected_img = cv.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))
        return corrected_img
    return False

def detectshot():

    arduino = serial.Serial('com12', 9600)
    print('Openconnection to Arduino')
    print('')
    arduino_data = arduino.readline()
    print(arduino_data)
    decoded_values = str(arduino_data[0:len(arduino_data)].decode("utf-8"))

    if '1' in decoded_values:

        im1 = get_frame()
        arduino_data = 0
        arduino.close()
        print('Connection closed')
        return im1

    elif '2' in decoded_values:
        im2 = get_frame()
        arduino_data = 0
        arduino.close()
        print('Connection closed')
        return im2

    else:
        pass

#
# def job():
#
#     im = detectshot()
#
#     diff, contour, img_detected = detect_arrow(im1, im2)
#
#     view_image(img_detected, 'Detektierung')
#
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def detect_arrow(img1, img2):

    # color correction - matplotlib and cv2 use different channels
    img1, img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB), cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    new_img = corrected(img2, img1)
    #
    # if new_img is not False:
    #     view_image(new_img, 'corr')

    diff = cv.absdiff(img1, new_img)

    height, width, channels = diff.shape
    print(width)
    print(height)
    print(channels)

    grayscaled = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(grayscaled, 5)
    retval, th = cv.threshold(blur, 12, 255, cv.THRESH_BINARY)

    # blank = np.zeros(shape=[height, width, 3], dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(th, kernel, iterations=1)
    new = cv.dilate(erosion, kernel, iterations=1)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv.morphologyEx(new, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(closing.copy(),
                                          cv.RETR_LIST,
                                          cv.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        if 700 < len(contours[i]) < 2500:
            print('##############')
            print(len(contours[i]))
            print('##############')
            erg = contours[i]

            img_contour = cv.drawContours(img2, contours[i], -1, (0, 255, 0), 5)

            a = erg.min(axis=0)
            b = np.where(erg == a[0][0])
            c = list(zip(b[0], b[1], b[2]))

            img_detected = img2.copy()
            for cord in c:
                if cord[2] == 0:
                    print(cord)
                    print(erg[cord[0], cord[1], 0])
                    print(erg[cord[0], cord[1], 1])

                    img_contour = cv.drawMarker(img2, (erg[cord[0], cord[1], 0], erg[cord[0], cord[1], 1]), color=(0, 0, 255), markerType=cv.MARKER_CROSS, thickness=10)

            # rows, cols = img2.shape[:2]
            # [vx, vy, x, y] = cv.fitLine(contours[i], cv.DIST_L2, 0, 0.01, 0.01)
            # slope = -float(vy) / float(vx)  # slope of the line
            # lefty = int((x * slope) + y)
            # righty = int(((x - cols) * slope) + y)
            # cv.line(img_detected, (cols - 1, righty), (0, lefty), (0, 255, 0), 5)

            # text1 = 'Aspect Ration: ' + str(round(aspect_ratio, 4))
            # text2 = 'Extent:  ' + str(round(extent, 4))
            # text3 = 'Solidity: ' + str(round(solidity, 4))
            # cv2.putText(img1, text1, (10, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
            # cv2.putText(img1, text2, (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
            # cv2.putText(img1, text3, (10, 90), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)

            # x, y, w, h = cv.boundingRect(contours[i])
            # area = cv.contourArea(contours[i])
            # Aspect_ratio = float(w) / h  # aspect ratio
            # rect_area = w * h
            # Extent = float(area) / rect_area
            # hull = cv.convexHull(contours[i])
            # hull_area = cv.contourArea(hull)
            # Solidity = float(area) / hull_area
            # cv.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 10)

            # text1 = 'Aspect Ration: ' + str(round(aspect_ratio, 4))
            # text2 = 'Extent:  ' + str(round(extent, 4))
            # text3 = 'Solidity: ' + str(round(solidity, 4))
            # cv2.putText(img1, text1, (10, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
            # cv2.putText(img1, text2, (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
            # cv2.putText(img1, text3, (10, 90), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)

        else:
            pass

    return diff, img_contour, img_detected


if __name__ == '__main__':

    ############################################# Test Arduino

    # print('Program started')
    #
    # # Setting up the Arduino
    # schedule.every(0.5).seconds.do(job)
    #
    # while True:
    #     schedule.run_pending()

    ############################################# Test correct() / absdiff()

    img1 = get_IMAGE('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_11.png', 0)
    img2 = get_IMAGE('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_12.png', 0)
    view_image(img1, 'Konturbild')
    view_image(img2, 'Konturbild')


    diff, contour, img_detected = detect_arrow(img1, img2)

    # view_image(th, 'th')
    # view_image(new, 'new')
    # view_image(blank, 'blank')
    # view_image(diff, 'Differenzbid')
    view_image(contour, 'Konturbild')
    view_image(img_detected, 'Detektierung')

    cv.waitKey(0)
    cv.destroyAllWindows()

    #get_VIDEO('/Users/alex/Workspace/git_repos/IBV-Projekt/Testvideos/271220/20201227_155744.mp4')

    # bild1 = test_einlesen.get_IMAGE('bilder/20201208_141922.jpg', 0)
    # bild2 = test_einlesen.get_IMAGE('bilder/20201208_141923.jpg', 0)
    #
    # dif_thred, grey1, grey2, diff = computeDifference(bild1, bild2)
    #
    # title = ['grey1', 'grey2', 'dif_thred', 'diff']
    # bilder = [dif_thred, grey1, grey2, diff]
    #
    # for k, i in enumerate(bilder):
    #     cv.namedWindow(title[k], cv.WINDOW_NORMAL)
    #     cv.moveWindow(title[k], 20, 20)
    #     cv.resizeWindow(title[k], 600, 600)
    #     cv.imshow(title[k], i)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()


#     final, combined, thresh, closing, frame_threshed_red_new, frame_threshed_green_new = detectDartboard(test_einlesen.get_IMAGE('bilder/20201208_141922.jpg', -1))
#     title = ['final', 'combined', 'thresh', 'closing', 'frame_threshed_red_new', 'frame_threshed_green_new']
#     #bilder = [combined, thresh, closing, frame_threshed_red_new, frame_threshed_green_new]
#     bilder = [final, combined]
#
#     for k, i in enumerate(bilder):
#         cv.namedWindow(title[k], cv.WINDOW_NORMAL)
#         cv.moveWindow(title[k], 20, 20)
#         cv.resizeWindow(title[k], 600, 600)
#         cv.imshow(title[k], i)
#
#     cv.waitKey(0)
#     cv.destroyAllWindows()