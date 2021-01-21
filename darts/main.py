from game import Game
from game import Board
import cv2 as cv
import time
import sys

if __name__ == '__main__':

    board = Board()
    board.set_ref_img()
    board.get_ref_img()

    while True:
        if board.detect_shot():

            board.set_img()
            board.get_img()

            diff, img_contour, img_detected = board.detect_arrow()

            board.view_image(diff, 'Diff')

            board.view_image(img_contour, 'Contours')

            board.view_image(img_detected, 'Detected')

            board.scorer()

        cv.waitKey(1)
        cv.destroyAllWindows()

        time.sleep(1)



    # game = Game()
    # while game.hasWinner() is False:
    #     for player in game.players:
    #         player.turn(game)
    #         if game.hasWinner() is True:
    #             break
