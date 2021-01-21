from game import Game
from game import Board
import cv2 as cv
import time
import sys

if __name__ == '__main__':

    print('Program started')

    board = Board('http://192.168.43.1:8080/shot.jpg', 'com7')

    board.set_ref_img()
    # board.get_ref_img()

    board.set_img()
    # board.get_img()

    board.ausgaben_text = 'Kalibration'

    cv.waitKey(1)
    cv.destroyAllWindows()

    board.ell_center = [(640, 480),
                        (640, 480),
                        (640, 480),
                        (640, 480),
                        (640, 480),
                        (640, 480)]

    board.ell_rad = [(20, 20),
                     (40, 40),
                     (100, 100),
                     (120, 120),
                     (380, 380),
                     (400, 400)]

    board.zone_center = (640, 480)
    board.zone_length = 400
    # self.zone_angle_offset = None
    board.zone_angle = [10,
                        33,
                        52,
                        69,
                        83,
                        96,
                        111,
                        126,
                        145,
                        167,
                        191,
                        213,
                        232,
                        248,
                        262,
                        276,
                        290,
                        306,
                        325,
                        346]

    db = board.draw_board()
    board.view_image(db, 'digBoard')

    cv.waitKey(1)
    cv.destroyAllWindows()
