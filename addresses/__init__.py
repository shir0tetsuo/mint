import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import uuid
import random
from typing import Optional

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

    def _gradient_colormap(self, cmap, n=10):
        return mcolors.LinearSegmentedColormap.from_list(
            cmap, self.maps[cmap], N=n
        )
    
    def colormap(self, cmap, n=10):

        if cmap not in self.maps:
            raise ValueError(f"Colormap '{cmap}' not found.")
        
        if len(self.maps[cmap]) == n:
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

class AddressHandler:
    def __init__(
            self, 
            base_directory=os.getcwd(), 
            glyph_subfolder='glyphtables', 
            color_subfolder='colors'
        ):

        self.glyphs = Glyphs(base_directory, subfolder=glyph_subfolder)
        self.colors = Colors(base_directory, subfolder=color_subfolder)

    def _rgba_to_hex(self, col):
        """Accepts (r,g,b) or (r,g,b,a) with floats in 0..1 or ints in 0..255.
        Returns '#rrggbb' (alpha ignored)."""
        # handle numpy arrays or similar
        if hasattr(col, "tolist"):
            col = col.tolist()

        if not isinstance(col, (tuple, list)):
            raise TypeError(f"Unsupported color type: {type(col)}")

        # Extract r,g,b (ignore alpha if present)
        if len(col) < 3:
            raise ValueError(f"Color must have at least 3 components: {col!r}")
        r, g, b = col[0], col[1], col[2]

        def to_byte(v):
            # floats in [0,1] -> byte; ints assumed 0..255
            if isinstance(v, float) and 0.0 <= v <= 1.0:
                return int(round(v * 255))
            return int(round(v))

        r_b, g_b, b_b = map(to_byte, (r, g, b))
        return "#{:02x}{:02x}{:02x}".format(r_b, g_b, b_b)

    def _normalize_hex(self, s):
        """Normalize hex like '#abc' or 'abc' or '#aabbcc' -> '#aabbcc' (lowercase)."""
        s = s.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            raise ValueError(f"Invalid hex color: {s!r}")
        return "#" + s.lower()

    def new_seed(self):
        '''Generates a new random seed.'''
        return str(uuid.uuid4())
    
    def table_from_seed(
            self,
            seed: Optional[str] = None,
            glyphs: str = 'Math1',
            colors: str = 'Beachgold',
            n: Optional[int] = None
        ):

        # Validate inputs
        glyphs = self.glyphs.maps.get(glyphs)[0] or self.glyphs.maps['Math1']

        if colors not in self.colors.maps:
            colors = 'Beachgold'

        n = n or len(self.colors.maps[colors])
        cmap = self.colors.colormap(colors, n=n)

        # Build a color source of (at least) up to n colors from cmap
        if hasattr(cmap, "colors") and getattr(cmap, "colors") is not None and len(getattr(cmap, "colors")) >= n:
            color_list = list(cmap.colors)[:n]
        else:
            if n == 1:
                color_list = [cmap(0.0)]
            else:
                color_list = [cmap(i / (n - 1)) for i in range(n)]

        # Basic validation
        if len(glyphs) == 0:
            raise ValueError("No glyphs available to choose from.")
        if len(color_list) == 0:
            raise ValueError("No colors available to choose from.")

        # Seed handling — ensure deterministic output
        if seed is None:
            seed = self.new_seed()
            print('New Table Seed:', seed)
        random.seed(seed)

        # Deterministic selection + padding by cycling if necessary
        import itertools

        # glyphs: pick up to n unique; if fewer, take all (shuffled) then cycle to n
        k_glyphs = min(n, len(glyphs))
        seed_glyphs = random.sample(glyphs, k_glyphs)
        if len(seed_glyphs) < n:
            seed_glyphs = list(itertools.islice(itertools.cycle(seed_glyphs), n))

        # colors: pick up to n unique; if fewer, take all (shuffled) then cycle to n
        k_colors = min(n, len(color_list))
        seed_colors = random.sample(color_list, k_colors)
        if len(seed_colors) < n:
            seed_colors = list(itertools.islice(itertools.cycle(seed_colors), n))

        if seed_colors:
            first = seed_colors[0]
            if isinstance(first, (tuple, list)):
                # tuple/list of numbers -> convert every entry to hex
                seed_colors = [self._rgba_to_hex(c) for c in seed_colors]
            elif isinstance(first, str):
                # assume hex-like strings; normalize them
                seed_colors = [self._normalize_hex(c) for c in seed_colors]
            else:
                # last-resort: try matplotlib's to_hex (handles many color formats)
                import matplotlib.colors as mcolors
                seed_colors = [mcolors.to_hex(c) for c in seed_colors]

        # Return exactly n entries (0..n-1)
        return {
            i: {
                'glyph': seed_glyphs[i],
                'color': seed_colors[i]
            }
            for i in range(n)
        }
        
    def generate_address(
            self, 
            seed:Optional[str]=None, 
            cols:int=9, 
            rows:int=1
        ):

        

        return