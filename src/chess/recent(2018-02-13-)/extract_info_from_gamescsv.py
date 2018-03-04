import csv

import os

os.chdir(os.path.dirname(__file__))
print(dir())
print(os.path.dirname(__file__))

"""
with open("/res/downloads/games.csv", newline="") as csvfile:
    games = csv.reader(csvfile, delimiter=",")
    for row in games:
        print(row)
"""