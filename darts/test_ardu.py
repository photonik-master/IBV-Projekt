import serial
from game import Board
import cv2 as cv
import sys
import time

board = Board('/dev/cu.usbmodem641', 'http://192.168.43.1:8080/shot.jpg')
board.set_ref_img()
# board.get_ref_img()
# cv.waitKey(1)
# cv.destroyAllWindows()

while True:
    if board.detect_shot():
        board.point = (0, 0)
        board.point_new = []

        # board.get_ref_img()
        # cv.waitKey(1)
        # cv.destroyAllWindows()

        # board.set_img()
        # board.get_img()
        # cv.waitKey(1)
        # cv.destroyAllWindows()

        bilder = board.detect_arrow()

        zahl = board.scorer()
        board.text_output = str(zahl)
        print(zahl)

        db = board.draw_board()
        board.view_image(db, 'Digital Board')
        cv.waitKey(1)
        cv.destroyAllWindows()

        board.set_ref_img()
        cv.waitKey(1)
        cv.destroyAllWindows()

    cv.waitKey(1)
    cv.destroyAllWindows()