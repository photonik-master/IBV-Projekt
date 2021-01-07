from game import Game
from game import Board
import cv2 as cv
import matplotlib as plt
import time
# import schedule
# import test

def view_image(im, name):
    # flipped = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    # gray = cv.cvtColor(flipped, cv.COLOR_BGR2GRAY)
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    # cv.moveWindow(name, 20, 20)
    cv.resizeWindow(name, 400, 400)
    cv.imshow(name, im)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':

    print('Program started')

    # Objekt Board wird erzeugt
    board = Board()

    # hier will ich die Sensorabfrage board.detect_shot() in eine Dauerschleife packen.
    # Das ensprechende Bild wird im board-Objekt gespeichert und kann weiter verarbeitet werden
    # je nach Rückgabe-Wert von board.detect_shot() (also True/False) wird weitergerechnet ...oder auch nicht
    while True:

        if board.detect_shot():
            print('shot!')
            view_image(board.get_img(), 'neues Bild')

            # board.detect_arrow()
        else:
            print('knopf!')
            view_image(board.get_ref_img(), 'neues Referenzbild')

        time.sleep(1)










    # für später, um Spiel zu starten
    # game = Game()
    # while game.hasWinner() is False:
    #     for player in game.players:
    #         player.turn(game)
    #         if game.hasWinner() is True:
    #             break






    # nur für den Tests!!! (Bilder einlesen)
    # board.img = board.get_image_file('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_22.png', 0)
    # board.ref_img = board.get_image_file('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_21.png', 0)

    # Die Differenz zwischen Bildern wird berechnet (Die Korrektur wird bem Aufruf durchgeführt)
    # diff, img_contour, img_detected = board.detect_arrow()

    # nur für den Tests!!! (Bilder anzeigen)
    # view_image(board.ref_img, 'Referenzbild')
    # view_image(board.img, 'Neues Bild')
    # view_image(diff, 'Differenzbild')
    # view_image(img_contour, 'Konturbild')
    # view_image(img_detected, 'Detektierung')
    # cv.waitKey(0)
    # cv.destroyAllWindows()
