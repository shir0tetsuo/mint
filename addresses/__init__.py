import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import uuid

def read_file_as_list(file_path):
    '''Returns list of lines from file (UTF-8).'''
    with open(file_path, 'r', encoding='UTF-8') as file:
        return [line.strip() for line in file]

def read_file_as_string(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        return f"Error: {e}"

class PathMap:
    def __init__(self, base_directory:str, subfolder:str):
        self.path = os.path.join(base_directory, subfolder)
        self.__items = [item for item in next(os.walk(self.path))[2]]
    
    @property
    def items(self):
        return self.__items


class Colors(PathMap):

    def __init__(self, base_directory, subfolder='colors'):
        super().__init__(base_directory, subfolder)

    @property
    def maps(self):
        return {
            os.path.splitext(item)[0]:
            read_file_as_list(os.path.join(self.path, item)) 
            for item in self.items
        }
    
    def _custom_colormap(self, cmap):
        return mcolors.ListedColormap(
            self.maps[cmap], name=cmap
        )

    def _gradient_colormap(self, cmap, n=256):
        return mcolors.LinearSegmentedColormap.from_list(
            cmap, self.maps[cmap], N=n
        )
    
    def colormap(self, cmap, n=256):
        '''
        Returns a colormap from the given name.
        
        If the colormap has 10 colors, it is treated as a custom colormap.
        If it has any more or less colors, it is treated as a gradient colormap.
        '''
        if cmap not in self.maps:
            raise ValueError(f"Colormap '{cmap}' not found.")
        
        if len(self.maps[cmap]) == 10:
            return self._custom_colormap(cmap)
        else:
            return self._gradient_colormap(cmap, n=n)

class Glyphs(PathMap):

    def __init__(self, base_directory, subfolder='glyphtables', fontdir='/usr/share/fonts/truetype/noto/'):
        super().__init__(base_directory, subfolder=subfolder)
        self.fontdir = fontdir
    
    @property
    def maps(self)-> dict[str, list[str|int]]:
        '''
        Reads glyphs and fonts from files.
            
            >>> {
                os.path.splitext(`filename`)[0]:
                [
                    `glyphs`,
                    `font_path`,
                    `font_size`
                ]
            }
        '''
        return {
            os.path.splitext(filename)[0]: [
                filedata[1],                             # Glyphs
                os.path.join(self.fontdir, filedata[0]), # Directory
                int(filedata[2])                         # Font Size (Generative Iter)
            ]
            for filename in self.items
            for filedata in [read_file_as_list(os.path.join(self.path, filename))]
        }
    