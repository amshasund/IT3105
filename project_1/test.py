import random


class Player:
    def __init__(self, env):
        self.name = "Tom"
        self.env = env

    def get_court_lengt(self):
        return self.env.get_lengt()


class Court:
    def __init__(self):
        self.length = random.randint(10, 50)

    def get_lengt(self):
        return self.length


class Team:
    def __init__(self):
        self.court = Court()
        self.player = Player(self.court)
        self.name = "Bla"

    def print_length(self):
        print(self.court.get_lengt())
        print(self.player.get_court_lengt())


# team = Team()
# print(team.name)
# print(team.court.length)
# print(team.player.env.length)

# team1 = Team()
# team1.print_length()

# team2 = Team()
# team2.print_length()

# team3 = Team()
# team3.print_length()


blabla = dict()
words = ["katt", "hund", "gris", "prinsesse"]
knots = ["bein", "hode", "armer"]
for w in words:
    blabla[w] = dict()
    for k in knots:
        blabla[w][k] = random.randint(0, 4)

print(blabla)

for key in blabla:
    blabla[key] = dict.fromkeys(blabla[key], 0)

print(blabla)
