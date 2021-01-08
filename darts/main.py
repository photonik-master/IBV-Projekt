from game import Game
from game import Board
import cv2 as cv
import time

if __name__ == '__main__':

    print('Program started')

    # Board-Objekt wird erzeugt
    board = Board()

    board.set_img()
    board.get_img()

    cv.waitKey(1)
    cv.destroyAllWindows()

    board.draw_board()

    cv.waitKey(1)
    cv.destroyAllWindows()

    # while True:
    #
    #     if board.detect_shot():
    #         print('shot!')
    #         print(board.get_img())
    #
    #         # board.detect_arrow()
    #     else:
    #         print('knopf!')
    #         print(board.get_ref_img())
    #
    #     time.sleep(1)

    cv.waitKey(5000)
    cv.destroyAllWindows()










    # für später, um Spiel zu starten
    # game = Game()
    # while game.hasWinner() is False:
    #     for player in game.players:
    #         player.turn(game)
    #         if game.hasWinner() is True:
    #             break
