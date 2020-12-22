from game import Game

if __name__ == '__main__':
    game = Game()

    while game.hasWinner() is False:
        for player in game.players:
            player.turn(game)
            if game.hasWinner() is True:
                break