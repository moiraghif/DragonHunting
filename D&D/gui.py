import constants
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk


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
            constants.ENEMY1_CHAR: self.load_image("enemy1.png"),
            constants.ENEMY2_CHAR: self.load_image("enemy2.png")
        }
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

    def move_agent(self, agent, pos):
        "move an agent on the screen"
        self.draw.move(self.players[agent.char],
                       pos[1] * self.size,
                       pos[0] * self.size)

    def start(self, iterations=10000):
        for p in self.world.players.keys():
            pos0 = self.world.players[p]
            pos, reward, done = p.get_action()
            delta = pos - pos0
            if all(delta == 0):
                rect = self.draw.create_rectangle(self.size * pos[1],
                                                  self.size * pos[0],
                                                  self.size * (pos[1] + 1),
                                                  self.size * (pos[0] + 1),
                                                  width=0, fill='yellow')
                time.sleep(0.07)
                self.draw.delete(rect)
            else:
                self.move_agent(p, self.world.players[p] - pos0)
                time.sleep(0.07)
        self.start(iterations - 1)
