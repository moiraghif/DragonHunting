import threading
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import constants

class GUI:
    def __init__(self, w, title="Dragon Hunting", size=700):
        self.world = w
        self.dim_y, self.dim_x = self.world.world.shape
        self.root = tk.Tk()
        self.size = round(min(map(lambda x: size/x, self.world.world.shape)))
        self.root.title(title)
        self.root.geometry("{}x{}".format(
            int(self.dim_x * self.size),
            int(self.dim_y * self.size)))
        self.root.resizable(False, False)
        self.draw = tk.Canvas(bg="white")
        self.draw.pack(fill=tk.BOTH, expand=1)
        self.pictures = {
            constants.WALL_CHAR: self.load_image("wall.png"),
            constants.PLAYER1_CHAR: self.load_image("player1.png"),
            constants.PLAYER2_CHAR: self.load_image("player2.png"),
            constants.DRAGON_CHAR: self.load_image("dino.png")
        }
        self.blood = self.load_image("blood.png")
        self.rip = self.load_image("rip.png")
        self.players = dict()
        self.refresh()
        threading.Thread(target=self.start).start()
        self.root.mainloop()

    def load_image(self, image):
        "return the image of the right size"
        return ImageTk.PhotoImage(
            Image.open("../icons/" + image).resize((self.size, self.size),
                                                   Image.ANTIALIAS))

    def refresh(self):
        "refresh the screen"
        self.draw.delete("all")
        for y in range(self.dim_y):
            self.draw.create_line(0, self.size * y,
                                  self.size * self.dim_x, self.size * y,
                                  fill="light grey")
        for x in range(self.dim_x):
            self.draw.create_line(self.size * x, 0,
                                  self.size * x, self.size * self.dim_x,
                                  fill="light grey")
        for x in range(self.dim_x):
            for y in range(self.dim_y):
                p = self.world.world[y, x]
                if p:
                    self.insert_picture(p, x, y)

    def insert_picture(self, p, x, y):
        "insert the picture"
        if p in self.pictures.keys():
            self.players[p] = self.draw.create_image(x * self.size,
                                                     y * self.size,
                                                     anchor=tk.NW,
                                                     image=self.pictures[p])

    def move_agent(self, agent, delta):
        "move an agent on the screen"
        self.draw.move(self.players[agent.char],
                       delta[1] * self.size,
                       delta[0] * self.size)

    def start(self, iterations=200):
        TIME = 0.1
        alive = {p: True for p in self.world.players.keys()}
        done=False
        for epoch in range(iterations):
            game_ended=False
            for p in alive.keys():
                if done:
                    game_ended=True
                    break
                if p.healt == 0:
                    alive[p] = False
                    reward, done = p.get_action()
                    if reward == -10:
                        pos = (self.world.players[p] * self.size)[::-1]
                        self.draw.delete(self.players[p.char])
                        self.draw.create_image((*pos),
                                               anchor=tk.NW,
                                               image=self.rip)
                    continue
                if game_ended:
                    break
                pos = self.world.players[p]
                healts0 = {p: p.healt for p in self.world.players.keys()}
                reward, done = p.get_action()
                delta_pos = self.world.players[p] - pos
                healts = {p: p.healt for p in self.world.players.keys()}
                if all(delta_pos == 0):
                    for agent, healt in healts.items():
                        if healts0[agent] - healt > 0 and healt > 0:
                            o_pos = (self.size * self.world.players[agent])
                            blood = self.draw.create_image((*o_pos[::-1]),
                                                           anchor=tk.NW,
                                                           image=self.blood)
                            time.sleep(TIME)
                            self.draw.delete(blood)
                else:
                    self.move_agent(p, delta_pos)
                    time.sleep(TIME)
            what_to_print = ""
            for p in self.world.players.keys():
                health = '0'+str(p.healt) if p.healt < 10 else str(p.healt)
                what_to_print = what_to_print + p.char + ":" + health + "\t"
            print(what_to_print, end='\r')
            if np.sum(np.array([v for k, v in alive.items()])) == 1:
                return
        print(what_to_print)
        return
