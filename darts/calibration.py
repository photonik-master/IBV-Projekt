import board
import cv2 as cv
import numpy as np


def calibration():
    ell_angle = [0, 0, 0, 1, 2, 2]
    ell_center = [(540, 850), (540, 848), (526, 839), (526, 839), (505, 835), (503, 832)]
    ell_rad = [(13, 17), (29, 37), (173, 214), (192, 235), (285, 354), (310, 379)]
    zone_center = (540, 850)
    zone_length = 500
    zone_angle = [13, 34, 52, 68, 85, 100, 115, 132, 152, 173, 196, 216, 235, 251, 265, 280, 295, 312, 330, 351]

    return ell_angle, ell_center, ell_rad, zone_center, zone_length, zone_angle


if __name__ == '__main__':
    print('Calibration started')

    board = board.Board('/dev/cu.usbmodem641', 'http://192.168.43.1:8080/video')

    board.set_ref_img()
    # board.get_ref_img()
    # cv.waitKey(1)
    # cv.destroyAllWindows()

    board.set_img()
    # board.get_img()
    # cv.waitKey(1)
    # cv.destroyAllWindows()

    board.text_output = 'Kalibration'

    db = board.draw_board()
    board.view_image(db, 'digBoard')
    cv.waitKey(1)
    cv.destroyAllWindows()
    #cv.imwrite('Kalibration_1.png', db)


