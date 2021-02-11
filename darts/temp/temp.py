from temp.temp import checkouts


class Game:
    winner = None
    type = 501
    players = []
    board = None

    def __init__(self):
        self.setUp()

    def setUp(self):
        print("Willkommen beim FH-Dart-Club")

        print("Welches Spiel? (301 / 501)")
        answer = int(input())
        self.setType(answer)

        print("Wie viele Spieler?")
        answer = int(input())
        self.setPlayers(answer)
        self.setBoard()
        return self

    def setPlayers(self, playerCount):
        players = []
        for i in range(1, playerCount + 1):
            print("Spieler {0}, wie ist dein Name?".format(i))
            name = input()
            players.append(Player(name, self.type))
            print("Hello " + name)
        self.players = players

    # TODO: Board Initialisierung
    def setBoard(self):
        self.board = Board()

    def setType(self, type):
        self.type = type

    def end(self):
        print("The Winner is %s" % self.winner.name)
        print("Game Over.")

    def hasWinner(self):
        if self.winner is not None:
            self.end()
            return True
        return False

    def hasOutshot(self, score):
        if str(score) in checkouts.checkouts():
            outshot = ''
            for shot in checkouts.checkouts()[str(score)]:
                outshot += shot + " "
            print("Outshot: {0}".format(outshot))
        else:
            print("Du benötigst noch: {0}".format(score))

class Player:
    tmpScore = None
    turnTotal = 0
    game = None

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def name(self):
        return self.name

    def updateScore(self):
        self.score -= self.turnTotal

    def turn(self, game):
        self.game = game
        self.resetTurnTotal()
        print("{0} ist am Zug, du hast {1} Punkten".format(self.name, self.score))
        self.tmpScore = self.score

        for dart in range(1, 4):
            self.throwDart(dart)

            if self.hasBust():
                return

            if self.hasWon():
                return

        print("{0} hat {1} Punkte erziehlt".format(self.name, self.turnTotal))
        print('')
        self.updateScore()

    def throwDart(self, dart):
        self.hasOutshot().setTurnTotal(self.getDartScore(dart))
        print("Erziehlt : {0}".format(self.turnTotal))

    def setTurnTotal(self, dartScore):
        self.turnTotal += dartScore

    def resetTurnTotal(self):
        self.turnTotal = 0

    def hasBust(self):
        if (self.tmpScore - self.turnTotal == 1) or (self.tmpScore - self.turnTotal < 0):
            print("%s bust!" % self.name)
            return True

    def getDartScore(self, dart):

        print("Wurf {0} ".format(dart))
        score = int(input())

        while score > 60:
            print("Hey Phil Taylor, mehr als 60?! Nochmal...")
            score = int(input())

        return score

    def hasOutshot(self):
        if str(self.tmpScore) in checkouts.checkouts():
            outshot = ''
            for shot in checkouts.checkouts()[str(self.tmpScore)]:
                outshot += shot + " "
            print("Outshot: {0}".format(outshot))
        else:
            pass
            # print("Du benötigst noch {0}".format(self.tmpScore))

        return self

    def hasWon(self):
        if self.score - self.turnTotal == 0:
            self.game.winner = self
            return True