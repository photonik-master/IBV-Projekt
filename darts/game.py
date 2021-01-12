import checkouts
import cv2 as cv
import numpy as np
import serial


class Game:
    winner = None
    type = 501
    players = []

    def __init__(self):
        self.setUp()

    def setUp(self):
        print("Willkommen beim FH-Dart-Club")

        print("Welches Spiel? (301 / 501)")
        answer = int(input())
        self.setType(answer)

        print("Wie viele Spieler?")
        answer = int(input())
        self.setPlayers(answer)
        return self

    def setPlayers(self, playerCount):
        players = []
        for i in range(1, playerCount + 1):
            print("Spieler {0}, wie ist dein Name?".format(i))
            name = input()
            players.append(Player(name, self.type))
            print("Hello " + name)
        self.players = players

    def setType(self, type):
        self.type = type

    def end(self):
        print("The Winner is %s" % self.winner.name)
        print("Game Over.")

    def hasWinner(self):
        if self.winner is not None:
            self.end()
            return True
        return False

    def hasOutshot(self, score):
        if str(score) in checkouts.checkouts():
            outshot = ''
            for shot in checkouts.checkouts()[str(score)]:
                outshot += shot + " "
            print("Outshot: {0}".format(outshot))
        else:
            print("Du benötigst noch: {0}".format(score))


class Player:
    tmpScore = None
    turnTotal = 0
    game = None

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def name(self):
        return self.name

    def updateScore(self):
        self.score -= self.turnTotal

    def turn(self, game):
        self.game = game
        self.resetTurnTotal()
        print("{0} ist am Zug, du hast {1} Punkten".format(self.name, self.score))
        self.tmpScore = self.score

        for dart in range(1, 4):
            self.throwDart(dart)

            if self.hasBust():
                return

            if self.hasWon():
                return

        print("{0} hat {1} Punkte erziehlt".format(self.name, self.turnTotal))
        self.updateScore()

    def throwDart(self, dart):
        self.hasOutshot().setTurnTotal(self.getDartScore(dart))
        print("Erziehlt : {0}".format(self.turnTotal))

    def setTurnTotal(self, dartScore):
        self.turnTotal += dartScore

    def resetTurnTotal(self):
        self.turnTotal = 0

    def hasBust(self):
        if (self.tmpScore - self.turnTotal == 1) or (self.tmpScore - self.turnTotal < 0):
            print("%s bust!" % self.name)
            return True

    def getDartScore(self, dart):

        print("Wurf {0} ".format(dart))
        score = int(input())

        while score > 60:
            print("Hey Phil Taylor, mehr als 60?! Nochmal...")
            score = int(input())

        return score

    def hasOutshot(self):
        if str(self.tmpScore) in checkouts.checkouts():
            outshot = ''
            for shot in checkouts.checkouts()[str(self.tmpScore)]:
                outshot += shot + " "
            print("Outshot: {0}".format(outshot))
        else:
            pass
            # print("Du benötigst noch {0}".format(self.tmpScore))

        return self

    def hasWon(self):
        if self.score - self.turnTotal == 0:
            self.game.winner = self
            return True


class Board:

    def __init__(self):

        # Arduino
        # try:
        #     print('Openconnection to Arduino')
        #     print('')
        #     self.arduino = serial.Serial('com7', 9600)
        #
        # except Exception as err:
        #     print(err)

        self.ref_img = None
        self.img = None

        # digBoard_circle
        self.rad = [10, 15, 100, 110, 200, 210]
        self.cen = None
        self.line_length = 210
        self.cir_center = (200, 200)
        self.angle_offset = 9

        self.point = (840, 1160)
        self.point_new = []

        # digBoard_ellipse
        self.ellipse = np.arange(0, 6)
        self.ellipse_score = [50, 25, 1, 3, 1, 2]
        self.ell_center = [(840, 1160),
                           (840, 1160),
                           (810, 1155),
                           (810, 1155),
                           (770, 1155),
                           (762, 1152)]

        self.ell_rad = [(25, 30),
                        (55, 70),
                        (320, 430),
                        (360, 470),
                        (535, 710),
                        (570, 758)]

        self.zone_center = (840, 1160)
        self.zone_length = 800
        # self.zone_angle_offset = None
        self.zone_angle = [10, 33, 52, 69, 83, 96, 111, 126, 145, 167, 191, 213, 232, 248, 262, 276, 290, 306, 325, 346]

    def __del__(self):
        print('Closeconnection to Arduino')
        print('')
        self.arduino.close()

    def set_ref_img(self):

        try:
            print('Openconnection to Cam')
            print('')
            cam = cv.VideoCapture(1)
            ret_val, img = cam.read()
            # img = cv.flip(img, 1)
            self.ref_img = img
            cam.release()

        except Exception as err:
            print(err)

    def get_ref_img(self):
        self.view_image(self.ref_img, 'neues Bild')
        return self.ref_img

    def set_img(self):

        try:
            print('Openconnection to Cam')
            print('')
            cam = cv.VideoCapture(0)
            ret_val, img = cam.read()
            # img = cv.flip(img, 1)
            self.img = img
            cam.release()

        except Exception as err:
            print(err)

    def get_img(self):
        self.view_image(self.img, 'neues Bild')
        return self.img

    @staticmethod
    def view_image(im, name):
        # flipped = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        # gray = cv.cvtColor(flipped, cv.COLOR_BGR2GRAY)
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        # cv.moveWindow(name, 20, 20)
        cv.resizeWindow(name, 400, 400)
        cv.imshow(name, im)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # TODO: hier Punktenzaehlung-Gesamtablauf
    def scorer(self):
        pass

    # TODO: hier Ellipse-Auswertung
    def scorer_ellipse(self):

        for i in self.ellipse:
            print(i)
            if self.is_inside_ellipse(self.point, self.ell_center[i], self.ell_rad[i]) == True:
                pass
            else:

                print('Treffer!')
                print(self.ellipse_score[i])
                break

    # TODO: hier Sektoren-Auswertung
    def scorer_sector(self):
        pass

    def draw_board(self):

        image = self.ref_img.copy()
        for i in self.ellipse:
            cv.ellipse(image, self.ell_center[i], self.ell_rad[i], 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)

        # a = np.arange(0 + offset, 360 + offset, 18)
        for angle in self.zone_angle:
            x2 = int(self.zone_center[0] + self.zone_length * np.cos(np.radians(angle)))
            y2 = int(self.zone_center[1] + self.zone_length * np.sin(np.radians(angle)))
            cv.line(image, self.zone_center, (x2, y2), (255, 255, 255), 1, cv.LINE_8, 0)

        cv.drawMarker(image, self.point, color=(255, 255, 255), markerType=cv.MARKER_CROSS, thickness=3)

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

        arduino_data = self.arduino.readline()
        print(arduino_data)
        decoded_values = str(arduino_data[0:len(arduino_data)].decode("utf-8"))

        if '1' in decoded_values:

            self.set_ref_img()
            arduino_data = 0
            # self.arduino.close()
            print('neues Bild')
            # print('Connection closed')
            print('')
            return True

        elif '2' in decoded_values:

            self.set_img()
            arduino_data = 0
            # arduino.close()
            print('neues Referenzbild')
            # print('Connection closed')
            print('')
            return True

        else:
            print('etwas ist schiefgelaufen :-)')
            print('')
            return False

    def detect_arrow(self):

        img1, img2 = cv.cvtColor(self.ref_img, cv.COLOR_BGR2RGB), cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        new_img = self.get_corrected_img(img2, img1)

        if new_img is not False:
            print('Bild korregiert')
            # view_image(new_img, 'corr')
        else:
            print('Bild nicht korregiert')

        diff = cv.absdiff(img1, new_img)

        grayscale = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(grayscale, 5)
        ret, th = cv.threshold(blur, 12, 255, cv.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(th, kernel, iterations=1)
        new = cv.dilate(erosion, kernel, iterations=1)

        kernel = np.ones((5, 5), np.uint8)
        closing = cv.morphologyEx(new, cv.MORPH_CLOSE, kernel)

        contours, hierarchy = cv.findContours(closing.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        img_contour = img2.copy()
        for i in range(len(contours)):
            if 700 < len(contours[i]) < 2500:
                print('Kontur {0}: {1}'.format(i, len(contours[i])))
                erg = contours[i]

                img_contour = cv.drawContours(img_contour, contours[i], -1, (0, 255, 0), 5)

                a = erg.min(axis=0)
                b = np.where(erg == a[0][0])
                c = list(zip(b[0], b[1], b[2]))

                img_detected = img2.copy()
                for cord in c:
                    if cord[2] == 0:
                        # print(cord)
                        # print(erg[cord[0], cord[1], 0])
                        # print(erg[cord[0], cord[1], 1])
                        self.point_new.append((erg[cord[0], cord[1], 0], erg[cord[0], cord[1], 1]))

                        img_detected = cv.drawMarker(img_detected, (erg[cord[0], cord[1], 0], erg[cord[0], cord[1], 1]),
                                                     color=(0, 0, 255), markerType=cv.MARKER_CROSS, thickness=10)
            else:
                pass

        self.point = self.point_new[0]
        # TODO: Mittelwert von Punkten
        # xx = 0
        # yy = 0
        # for i in self.point_new:
        #     print(i)
        #     xx += i[0]
        #     yy += i[1]
        #
        # self.point = (xx/len(self.point_new), yy/len(self.point_new))

        return diff, img_contour, img_detected

    def is_inside_ellipse(self, point, center, rad):

        a = ((point[0] - center[0]) ** 2) / (rad[0] ** 2)
        b = ((point[1] - center[1]) ** 2) / (rad[1] ** 2)

        if (a + b) < 1:
            return False
        else:
            return True
