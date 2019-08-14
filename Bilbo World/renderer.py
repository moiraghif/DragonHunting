import math
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont
import IPython.display as IPdisplay

#font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
font = ImageFont.truetype('DejaVuSansMono.ttf', 90)

class Renderer():
    """renders a grid with values (for the gridworld)"""
    def __init__(self, grid, cell_size=60):
        self.grid = grid
        self.cell_size = cell_size

        grid_h = len(grid)
        grid_w = max(len(row) for row in grid)
        self.size = (grid_w * self.cell_size, grid_h * self.cell_size)

    def _draw_cell(self, x, y, fill, color, value, pos, text_padding=10):
        self.draw.rectangle([(x, y), (x+self.cell_size, y+self.cell_size)], fill=fill)

        # render text
        y_mid = math.floor(self.cell_size/2)
        lines = textwrap.wrap(str(value), width=15)
        _, line_height = self.draw.textsize(lines[0], font=font)
        height = len(lines) * line_height + (len(lines) - 1) * text_padding
        current_height = y_mid - height/2

        for line in lines:
            w, h = self.draw.textsize(line, font=font)
            self.draw.text((x + (self.cell_size - w)/2, y + current_height), line, font=font, fill=color)
            current_height += h + text_padding

    def render(self, pos=None):
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

        return self.img


n15 = np.load('qtable_15.npy')

world = np.array([["" for x in range(15)]
                      for y in range(15)])

np.random.seed(seed=1234)

for i in range(15):
    for j in range(15):
        if ((np.random.rand() > 1 - 0.12) and (world[i, j]=='')):
            world[i,j]='█'

world[round(15/2)-1,15-3]='☠'
world[round(15/2)-1,15-1]='♚'


for i in range(15):
    for j in range(15):
        if world[i,j] in ['☠','█','♚']:
            continue
        action = np.argmax(n15[i,j])
        if action==0:
            world[i,j]='⬆'
        elif action==1:
            world[i,j]='⬇'
        elif action==2:
            world[i,j]='⬅'
        elif action==3:
            world[i,j]='➡'

renderer = Renderer(world, cell_size=100)
renderer.render().save('PolicyMap.png')
