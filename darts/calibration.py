from game import Board
import cv2 as cv


def calibration():
    ell_angle = [0, 0, -1, -1, 0, -1],
    ell_center = [(607, 865), (607, 862), (582, 856), (578, 855), (542, 848), (536, 847)]
    ell_rad = [(20, 20), (44, 50), (256, 302), (283, 333), (434, 500), (464, 530)]
    zone_center = (607, 865)
    zone_length = [600]
    zone_angle_offset = None
    zone_angle = [14, 35, 53, 70, 86, 101, 117, 134, 153, 174, 195, 215, 233, 250, 266, 281, 297, 314, 333, 354]

    return ell_angle, ell_center, ell_rad, zone_center, zone_length, zone_angle


if __name__ == '__main__':
    print('Calibration started')

    board = Board('/dev/cu.usbmodem641', 'http://192.168.43.1:8080/video')

    ell_angle, ell_center, ell_rad, zone_center, zone_length, zone_angle = calibration()

    board.set_ref_img()
    # board.get_ref_img()
    # cv.waitKey(1)
    # cv.destroyAllWindows()

    board.set_img()
    # board.get_img()
    # cv.waitKey(1)
    # cv.destroyAllWindows()

    board.text_output = 'Kalibration'

    board.ell_angle = ell_angle
    board.ell_center = ell_center
    board.ell_rad = ell_rad

    board.zone_center = zone_center
    board.zone_length = zone_length
    board.zone_angle = zone_angle

    db = board.draw_board()
    board.view_image(db, 'digBoard')
    cv.waitKey(1)
    cv.destroyAllWindows()
