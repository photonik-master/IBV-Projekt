from game import Game
from game import Board
import cv2 as cv
import time
import sys

if __name__ == '__main__':
    print('Program started')

    board = Board()

    board.view_image(board.set_ref_img(), 'Ref_Img')
    cv.waitKey(0)
    cv.destroyAllWindows()

    board.ell_center = [(840, 1160),
                        (840, 1160),
                        (810, 1155),
                        (810, 1155),
                        (770, 1155),
                        (762, 1152)]

    board.ell_rad = [(5, 5),
                     (5, 5),
                     (5, 5),
                     (5, 5),
                     (5, 5),
                     (5, 5)]

    board.zone_center = (840, 1160)
    board.zone_length = 800
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
    cv.waitKey(0)
    cv.destroyAllWindows()
