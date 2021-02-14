from board import Game
from board import Board
import cv2 as cv
import time
import test

if __name__ == '__main__':
    print('Program started')

    board = Board('http://192.168.43.1:8080/shot.jpg', 'com7')

    board.ref_img = test.get_IMAGE('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_21.png', 0)
    board.get_ref_img()
    cv.waitKey(1)
    cv.destroyAllWindows()

    board.img = test.get_IMAGE('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_22.png', 0)
    board.get_img()
    cv.waitKey(1)
    cv.destroyAllWindows()

    bilder = board.detect_arrow()
    name = ['perspektive.png', 'diff.png', 'blur.png', 'th.png', 'erosion.png', 'dilation.png', 'closing.png', 'contour.png', 'detected.png']

    for i, k in enumerate(bilder):
        board.view_image(k, name[i])
        # cv.imwrite(name[i], k)

        cv.waitKey(1)
        cv.destroyAllWindows()

    zahl = board.scorer()
    board.ausgaben_text = str(zahl)
    print(zahl)

    db = board.draw_board()
    board.view_image(db, 'Digital Board')

    # cv.imwrite('digBoard.png', db)

    cv.waitKey(1)
    cv.destroyAllWindows()
