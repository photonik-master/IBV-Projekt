from game import Game
from game import Board
import cv2 as cv
import time
import sys
import sys

if __name__ == '__main__':

    print('Program started')

    board = Board()
    board.set_ref_img()

    db = board.draw_board()
    board.view_image(db, 'Digital Board')
    cv.waitKey(0)
    cv.destroyAllWindows()

    answer = input('Fortsetzen? (j/n)')
    if answer == 'J' or 'j':
        pass
    else:
        sys.exit(0)

    i = 0
    j = 0
    n = 0
    while True:

        if board.detect_shot():
            i += 1

            board.get_img()
            cv.waitKey(1000)
            cv.destroyAllWindows()

            diff, img_contour, img_detected = board.detect_arrow()

            board.view_image(diff, 'Differenz')
            cv.waitKey(1000)
            cv.destroyAllWindows()

            board.view_image(img_contour, 'Kontur')
            cv.waitKey(1000)
            cv.destroyAllWindows()

            board.view_image(img_detected, 'Pfeil')
            cv.waitKey(1000)
            cv.destroyAllWindows()

            zahl = board.scorer()
            board.ausgaben_text = str(zahl)

            db = board.draw_board()
            board.view_image(db, 'Digital Board')
            cv.waitKey(1000)
            cv.destroyAllWindows()

            answer = input('Richtig?')
            if answer == '':
                j += 1
            else:
                n += 1

            print("Fehlerquote: {0}".format((n * 100) / i))
            print('')

        else:
            pass

        time.sleep(1)
