import numpy as np
import os

MAZE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../mazes/"


class Maze:
    def __init__(self, name):
        self.name = str(name)
        self._height = None
        self._width = None
        self._wall_cells = None
        self._process_maze_file()

    def _process_maze_file(self):
        maze_path = MAZE_DIR + self.name + ".maze"
        with open(maze_path) as handle:
            data = handle.readlines()

        self._height = len(data)
        self._width = len(data[0].replace(" ", "").rstrip())

        self._wall_cells = []
        for row_id, row in enumerate(data):
            for col_id, cell in enumerate(row.replace(" ", "").rstrip()):
                if cell == "#":
                    self._wall_cells.append((row_id, col_id))

    def get_walls(self):
        rows, columns = list(zip(*self._wall_cells))
        return np.array(rows), np.array(columns)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width
