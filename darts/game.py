from calibration import calibration
import cv2 as cv
import numpy as np
import serial
import time
import sys


class Board:

    def __init__(self, com, url):

        try:
            self.ser = serial.Serial(com, 9600)
            print('Openconnection to Arduino')
        except serial.SerialException:
            print('Serial Port Number?')
            sys.exit(0)

        self.url = url

        self.text_output = ''
        self.ref_img = None
        self.img = None

        self.calibrate()

    def calibrate(self):

        ell_angle, ell_center, ell_rad, zone_center, zone_length, zone_angle = calibration()

        #
        # # digBoard_ellipse
        #
        # self.ellipse = np.arange(0, 6)
        #
        # self.ellipse_score = [50, 25, 1, 3, 1, 2]
        #
        # self.ell_angle = [0, 0, -1, -1, 0, -1]
        #
        # self.ell_center = [(607, 865),
        #                    (607, 862),
        #                    (582, 856),
        #                    (578, 855),
        #                    (542, 848),
        #                    (536, 847)]
        #
        # self.ell_rad = [(20, 20),
        #                 (44, 50),
        #                 (256, 302),
        #                 (283, 333),
        #                 (434, 500),
        #                 (464, 530)]
        #
        # self.zone_center = (607, 865)
        #
        # self.zone_length = 600
        #
        # # self.zone_angle_offset = None
        #
        # self.zone_angle = [15,
        #                    36,
        #                    54,
        #                    70,
        #                    86,
        #                    101,
        #                    117,
        #                    134,
        #                    153,
        #                    174,
        #                    195,
        #                    215,
        #                    233,
        #                    249.5,
        #                    266,
        #                    281,
        #                    297,
        #                    314,
        #                    333,
        #                    354]

        # digBoard_circle

        # self.rad = [10, 15, 100, 110, 200, 210]
        # self.cen = None
        # self.line_length = 210
        # self.cir_center = (200, 200)
        # self.angle_offset = 9
        #
        # self.point = (0, 0)
        # self.point_new = []


    def set_ref_img(self):

        cam = cv.VideoCapture(self.url)
        ret_val, img = cam.read()
        self.ref_img = img
        print('Referenzbild: {0}'.format(img.shape))
        cam.release()

    def get_ref_img(self):
        self.view_image(self.ref_img, 'Referenzbild')
        # return self.ref_img

    def set_img(self):

        self.text_output = ''

        cam = cv.VideoCapture(self.url)
        ret_val, img = cam.read()
        self.img = img
        print('Bild: {0}'.format(img.shape))
        cam.release()

    def get_img(self):
        self.view_image(self.img, 'Bild')
        # return self.img

    @staticmethod
    def view_image(im, name):
        # flipped = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        # gray = cv.cvtColor(flipped, cv.COLOR_BGR2GRAY)
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.moveWindow(name, 20, 20)
        cv.resizeWindow(name, 400, 400)
        cv.imshow(name, im)
        cv.waitKey(2000)
        cv.destroyAllWindows()

    def scorer(self):

        theta = None
        for i in self.ellipse:
            if self.is_inside_ellipse(self.point, self.ell_center[i], self.ell_rad[i]) == True:
                pass
            else:
                # print('Ellipse: {0}'.format(i))
                if self.ellipse_score[i] == 50 or self.ellipse_score[i] == 25:
                    print(self.ellipse_score[i])
                    return self.ellipse_score[i]
                else:
                    # print('Multiplikation mit: {0}'.format(self.ellipse_score[i]))
                    factor = self.ellipse_score[i]
                    dx = self.point[0] - self.zone_center[0]
                    dy = self.point[1] - self.zone_center[1]
                    # print(dx)
                    # print(dy)
                    if dy > 0 and dx > 0:
                        theta = np.degrees(np.arctan(dy / dx))
                        break
                    elif dy > 0 and dx < 0:
                        theta = 180 - np.degrees(np.arctan(dy / abs(dx)))
                        break
                    elif dy < 0 and dx < 0:
                        theta = 180 + np.degrees(np.arctan(abs(dy) / abs(dx)))
                        break
                    elif dy < 0 and dx > 0:
                        theta = 360 - np.degrees(np.arctan(abs(dy) / dx))
                        break
                    elif dx == 0 and dy < 0:
                        theta = 270
                        break
                    elif dx == 0 and dy > 0:
                        theta = 90
                        break
                    elif dy == 0 and dx < 0:
                        theta = 180
                        break
                    elif dy == 0 and dx > 0:
                        theta = 0
                        break
                    else:
                        print('scorer fehler!!!')

        if theta == None:
            return 'Nicht im Zaehlbereich'
        else:
            # [10, 33, 52, 69, 83, 96, 111, 126, 145, 167, 191, 213, 232, 248, 262, 276, 290, 306, 325, 346]
            po = [6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13]
            for i, l in enumerate(self.zone_angle):
                # print(i)
                # print(l)
                # print(punkte[i])
                if theta < l:
                    # print(po[i] * factor)
                    return po[i] * factor
                elif theta == l:
                    # TODO: In diesem Fall müssen die Punkte manuell eingetragen werden
                    pass
                elif self.zone_angle[19] < theta < 360:
                    # print(po[0])
                    return po[0]
                else:
                    pass

    def draw_board(self):

        image = self.img.copy()
        for i in self.ellipse:
            cv.ellipse(image, self.ell_center[i], self.ell_rad[i], self.ell_angle[i], 0, 360, (255, 255, 255), 1,
                       cv.LINE_8, 0)

        for angle in self.zone_angle:
            x2 = int(self.zone_center[0] + self.zone_length * np.cos(np.radians(angle)))
            y2 = int(self.zone_center[1] + self.zone_length * np.sin(np.radians(angle)))
            cv.line(image, self.zone_center, (x2, y2), (255, 255, 255), 1, cv.LINE_8, 0)

        cv.drawMarker(image, self.point, color=(255, 255, 255), markerType=cv.MARKER_CROSS, thickness=3)

        font = cv.FONT_HERSHEY_SIMPLEX
        org = (20, 100)
        fontScale = 4
        color = (255, 0, 0)
        thickness = 10
        cv.putText(image, self.text_output, org, font, fontScale, color, thickness, cv.LINE_AA)

        return image

    def get_corrected_img(self, img1, img2):  # perspektive von bild 2 wird auf bild 1 angepasst. (korrigiert)

        MIN_MATCHES = 50

        orb = cv.ORB_create(nfeatures=500)  # 500 features
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

    def detect_shot(self):

        ard = self.ser.readline()
        self.ser.flush()

        if ard == b'shot\r\n':
            print('shot')
            time.sleep(3)
            self.set_img()
            return True
        else:
            return False

    def detect_arrow(self):

        img1, img2 = cv.cvtColor(self.ref_img, cv.COLOR_BGR2RGB), cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        new_img = self.get_corrected_img(img2, img1)

        if new_img is not False:
            print('Bild korregiert')
        else:
            print('Bild nicht korregiert')

        # self.view_image(new_img, 'new_image')
        # cv.waitKey(1)
        # cv.destroyAllWindows()

        diff = cv.absdiff(img1, new_img)

        # self.view_image(diff, 'diff')
        # cv.waitKey(1)
        # cv.destroyAllWindows()

        grayscale = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(grayscale, 5)
        ret, th = cv.threshold(blur, 12, 255, cv.THRESH_BINARY)

        # self.view_image(th, 'th')
        # cv.waitKey(1)
        # cv.destroyAllWindows()

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(th, kernel, iterations=1)
        new = cv.dilate(erosion, kernel, iterations=1)

        kernel = np.ones((15, 15), np.uint8)
        closing = cv.morphologyEx(new, cv.MORPH_CLOSE, kernel)

        # self.view_image(closing, 'th')
        # cv.waitKey(1)
        # cv.destroyAllWindows()

        contours, hierarchy = cv.findContours(closing.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        img_contour = img2.copy()
        for i in range(len(contours)):
            if 500 < len(contours[i]) < 1500:
                print('Kontur {0}: {1}'.format(i, len(contours[i])))
                erg = contours[i]

                img_contour = cv.drawContours(img_contour, contours[i], -1, (0, 255, 0), 3)

                a = erg.min(axis=0)
                b = np.where(erg == a[0][0])
                c = list(zip(b[0], b[1], b[2]))

                img_detected = img2.copy()
                for cord in c:
                    if cord[2] == 0:
                        print(cord)
                        # print(erg[cord[0], cord[1], 0])
                        # print(erg[cord[0], cord[1], 1])
                        self.point_new.append((erg[cord[0], cord[1], 0], erg[cord[0], cord[1], 1]))

                        img_detected = cv.drawMarker(img_detected, (erg[cord[0], cord[1], 0], erg[cord[0], cord[1], 1]),
                                                     color=(0, 0, 255), markerType=cv.MARKER_CROSS, thickness=10)

                # self.view_image(img_detected, 'detected')
                # cv.waitKey(1)
                # cv.destroyAllWindows()

            else:
                pass

        xx = 0
        yy = 0
        k = 0
        for i in self.point_new:
            xx += i[0]
            yy += i[1]
            k += 1

        # TODO: k==0 kein Kontur
        self.point = (round(xx / k), round(yy / k))
        print('Auftreffpunkt: {0}'.format(self.point))

        # return [new_img, diff, blur, th, erosion, new, closing, img_contour, img_detected]
        # return diff, img_contour, img_detected

    def is_inside_ellipse(self, point, center, rad):
        a = ((point[0] - center[0]) ** 2) / (rad[0] ** 2)
        b = ((point[1] - center[1]) ** 2) / (rad[1] ** 2)

        if (a + b) < 1:
            return False
        else:
            return True
