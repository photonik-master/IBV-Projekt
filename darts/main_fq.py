from game import Game
from game import Board
import cv2 as cv
import time
import sys

if __name__ == '__main__':

    print('Program started')

    board = Board()
    board.set_ref_img()

    db = board.draw_board()
    board.view_image(db, 'Digital Board')
    cv.waitKey(0)
    cv.destroyAllWindows()

    while True:

        if board.detect_shot():

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

        else:
            pass

        time.sleep(1)










    # für später, um Spiel zu starten
    # game = Game()
    # while game.hasWinner() is False:
    #     for player in game.players:
    #         player.turn(game)
    #         if game.hasWinner() is True:
    #             break
