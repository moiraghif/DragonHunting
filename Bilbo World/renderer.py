import math
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont
import IPython.display as IPdisplay

#font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

class Renderer():
    """renders a grid with values (for the gridworld)"""
    def __init__(self, grid, cell_size=60):
        self.grid = grid
        self.cell_size = cell_size
        self.font = ImageFont.truetype('DejaVuSansMono.ttf', 90)

        grid_h = len(grid)
        grid_w = max(len(row) for row in grid)
        self.size = (grid_w * self.cell_size, grid_h * self.cell_size)

    def _draw_cell(self, x, y, fill, color, value, pos, text_padding=10):
        self.draw.rectangle([(x, y), (x+self.cell_size, y+self.cell_size)], fill=fill)
        # render text
        y_mid = math.floor(self.cell_size/2)
        lines = textwrap.wrap(str(value), width=15)
        _, line_height = self.draw.textsize(lines[0], font=self.font)
        height = len(lines) * line_height + (len(lines) - 1) * text_padding
        current_height = y_mid - height/2
        for line in lines:
            w, h = self.draw.textsize(line, font=self.font)
            self.draw.text((x + (self.cell_size - w)/2, y + current_height), line, font=self.font, fill=color)
            current_height += h + text_padding

    def render(self, episode=None, pos=None):
        """renders the grid,
        highlighting the specified position if there is one"""
        self.img = Image.new('RGBA', self.size, color=(255,255,255,0))
        self.draw = ImageDraw.Draw(self.img)


        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val is None:
                    continue
                fill = (220,220,220,255) if (r + c) % 2 == 0 else (225,225,225,255)

                # current position
                if pos is not None and pos == (r, c):
                    fill = (255,255,150,255)
                self._draw_cell(c * self.cell_size, r * self.cell_size, fill, (0,0,0,255), val, (r,c))
        if not episode is None:
            self.draw.text((0,0),'ep: '+str(episode), font = ImageFont.truetype('DejaVuSansMono.ttf', 30), fill = (0,0,0,255))
        return self.img

def render_world(world, WORLD_DIM, qtable, episode=None):

    world[round(WORLD_DIM/2)-1, WORLD_DIM-1] = '♚' #in caso sia sparito
    for i in range(WORLD_DIM):
        for j in range(WORLD_DIM):
            if world[i,j] in ['☠','█','♚']:
                continue
            action = np.argmax(qtable[i,j])
            if action==0:
                world[i,j]='⬆'
            elif action==1:
                world[i,j]='⬇'
            elif action==2:
                world[i,j]='⬅'
            elif action==3:
                world[i,j]='➡'

    renderer = Renderer(world, cell_size=100)
    return renderer.render(episode)

if __name__ == '__main__':
    from agents import QLearningAgent, WORLD_DIM
    from CreateBilboWorld import *
    import os
    import numpy as np

    bilbo = QLearningAgent(PLAYER_CHAR)
    mondo = World(WORLD_DIM, bilbo=bilbo, obstacle=True)
    qtable = np.load("./models/qtable_" + str(WORLD_DIM) + ".npy")
    img = render_world(mondo.world, WORLD_DIM, qtable)
    img.show()
