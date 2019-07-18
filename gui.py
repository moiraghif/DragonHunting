import CreateBilboWorld as world
import tkinter as tk
from PIL import Image, ImageTk


class GUI:
    def __init__(self, w, title="Dragon Hunting", size=700):
        self.world = w
        self.dim_y, self.dim_x = self.world.shape
        self.root = tk.Tk()
        self.size = round(min(map(lambda x: size/x, self.world.shape)))
        self.root.title(title)
        self.root.geometry("{}x{}".format(
            int(self.dim_x * self.size),
            int(self.dim_y * self.size)))
        self.root.resizable(False, False)
        self.draw = tk.Canvas(bg="white")
        self.draw.pack(fill=tk.BOTH, expand=1)
        self.pictures = {
            world.DRAGON_CHAR: self.load_image("./icons/dino.png"),
            world.OBSTACLE_CHAR: self.load_image("./icons/rock.png"),
            world.TREASURE_CHAR: self.load_image("./icons/treasure.png"),
            world.PLAYER_CHAR: self.load_image("./icons/man.png")
        }
        self.refresh()
        self.root.mainloop()

    def load_image(self, image):
        "return the image of the right size"
        return ImageTk.PhotoImage(
            Image.open(image).resize((self.size, self.size),
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
                p = self.world[y, x]
                if p:
                    self.insert_picture(p, x, y)

    def insert_picture(self, p, x, y):
        "insert the picture"
        if p in self.pictures.keys():
            self.draw.create_image(x * self.size,
                                   y * self.size,
                                   anchor=tk.NW,
                                   image=self.pictures[p])

    def insert_agent(self, agent, x=0, y=0):
        "move an agent on the screen"
        self.draw.move(agent,
                       x * self.size,
                       y * self.size)


if __name__ == "__main__":
    player = world.Agent(world.PLAYER_CHAR)
    world1 = world.World(bilbo=player,obstacle=True).world
    GUI(w=world1)
    for i in range(3):
        player.move(player.random_action())()
