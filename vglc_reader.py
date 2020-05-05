import json
import math
import os
from glob import glob
from enum import IntEnum
import numpy as np


class VGLCLevelRepresentationType(IntEnum):
    RAW = 0
    ONE_HOT = 1
    BINARY = 2
    SORTED = 3


class VGLCGameData:
    def __init__(self, json_path):
        json_obj = json.load(open(json_path))
        self.tile_info = json_obj['tiles']
        empty_keys = [k for k in self.tile_info.keys() if 'empty' in self.tile_info[k]]
        non_empty_keys = [k for k in self.tile_info.keys() if 'empty' not in self.tile_info[k]]
        self.sorted_tile_types = sorted(empty_keys) + sorted(non_empty_keys)
        self.tiles_to_indices = {x: i for i, x in enumerate(self.sorted_tile_types)}
        json_dir = os.path.split(json_path)[0]
        levels_dir = os.path.join(json_dir, 'Processed')
        self.levels_contents = []
        for level_file in glob(os.path.join(levels_dir, '*.txt')):
            level_contents = open(level_file).readlines()
            level_contents = [line.strip() for line in level_contents]
            self.levels_contents.append(level_contents)

    def __len__(self):
        return len(self.levels_contents)

    def get_level(self, index, representation_type: VGLCLevelRepresentationType = VGLCLevelRepresentationType.RAW) -> np.array:
        contents = self.levels_contents[index]
        level_data = []
        for line in contents:
            loaded_line = []
            for char in line:
                if representation_type == VGLCLevelRepresentationType.RAW:
                    loaded_line.append(char)
                elif representation_type == VGLCLevelRepresentationType.ONE_HOT:
                    one_hot = [0] * len(self.sorted_tile_types)
                    one_hot[self.tiles_to_indices[char]] = 1
                    loaded_line.append(one_hot)
                elif representation_type == VGLCLevelRepresentationType.BINARY:
                    is_empty = 'empty' in self.tile_info[char]
                    loaded_line.append(1 - int(is_empty))
                elif representation_type == VGLCLevelRepresentationType.SORTED:
                    loaded_line.append(self.tiles_to_indices[char])
            level_data.append(loaded_line)
        return np.array(level_data)

    def split_level_to_pages(self, level_array: np.array, overlap: int=0) -> np.array:
        height = level_array.shape[0]
        width = level_array.shape[1]
        pages = []
        i = 0
        while i + height < width:
            page = level_array[:, i:i+height]
            pages.append(page)
            if overlap >= 0:
                i += (height - overlap)
            else:
                i -= overlap
        return np.array(pages)




def __main__():
    import os
    p = os.path.join(os.curdir, '..', 'TheVGLC', 'Super Mario Bros', 'smb.json')
    game_data = VGLCGameData(p)
    for t in VGLCLevelRepresentationType:
        level = game_data.get_level(0, t)
        print(level.shape)
        pages = game_data.split_level_to_pages(level, 0)
        print(pages.shape)
        pages = game_data.split_level_to_pages(level, 5)
        print(pages.shape)
        pages = game_data.split_level_to_pages(level, -2)
        print(pages.shape)


if __name__ == '__main__':
    __main__()

