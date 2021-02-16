import board
import cv2 as cv
import time

if __name__ == '__main__':
    board = board.Board('/dev/cu.usbmodem641', 'http://192.168.43.1:8080/video')

    board.set_ref_img()

    i = 0
    while True:
        if board.detect_shot():
            time.sleep(1)
            if board.detect_arrow() != False:
                board.text_output = str(board.scorer())
                db = board.draw_board()
                board.view_image(db, 'digBoard')
                cv.waitKey(1)
                cv.destroyAllWindows()
                board.set_ref_img()
                i += 1
                if i == 3:
                    input('Nächste Runde?')
                    i = 0
                    board.set_ref_img()
                else:
                    pass
        else:
            print('Nicht erkannt!')
