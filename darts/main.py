from game import Game
from game import Board
import cv2 as cv
import time
import sys

if __name__ == '__main__':

    board = Board()
    board.set_ref_img()

    while True:
        if board.detect_shot():

            board.set_img()

            board.detect_arrow()

            board.scorer()



    # game = Game()
    # while game.hasWinner() is False:
    #     for player in game.players:
    #         player.turn(game)
    #         if game.hasWinner() is True:
    #             break
