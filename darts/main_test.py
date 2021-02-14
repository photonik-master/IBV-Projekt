import board
import cv2 as cv
import time
import sys

if __name__ == '__main__':
    board = board.Board('/dev/cu.usbmodem641', 'http://192.168.43.1:8080/video')

    board.set_ref_img()
    board.get_ref_img()

    cv.waitKey(1)
    cv.destroyAllWindows()

    board.set_img()
    board.get_img()

    cv.waitKey(1)
    cv.destroyAllWindows()

    board.detect_arrow()

    cv.waitKey(1)
    cv.destroyAllWindows()

    board.text_output = str(board.scorer())

    db = board.draw_board()
    board.view_image(db, 'digBoard')
    cv.waitKey(1)
    cv.destroyAllWindows()
