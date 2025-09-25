import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import uuid
import threading
import numpy as np
import json
import random
from typing import Optional, Any, Literal, Sequence, Union, Tuple
import numbers
import inspect
from PIL import Image, PngImagePlugin

def drop_trailing_integers(seq):
    if isinstance(seq, str):
        # Strip trailing digit characters
        i = len(seq)
        while i > 0 and seq[i-1].isdigit():
            i -= 1
        return seq[:i]
    elif isinstance(seq, (list, tuple)):
        i = len(seq)
        while i > 0 and isinstance(seq[i-1], int):
            i -= 1
        return type(seq)(seq[:i])
    else:
        return seq

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
        data = {
            os.path.splitext(filename)[0]: [
                filedata[1],                             # Glyphs
                os.path.join(self.fontdir, filedata[0]), # Directory
                int(filedata[2])                         # Font Size (Generative Iter)
            ]
            for filename in self.items
            for filedata in [
                read_file_as_list(os.path.join(self.path, filename))
            ]
        }

        return data
    

class Diacritics(PathMap):

    def __init__(self, base_directory, subfolder='diacritics'):
        super().__init__(base_directory, subfolder)

    @property
    def maps(self) -> dict[str, list[str]]:
        data = {
            os.path.splitext(filename)[0]: [
                filedata[0],                             # Diacritics for Glyphs
            ]
            for filename in self.items
            for filedata in [
                read_file_as_list(os.path.join(self.path, filename))
            ]
        }

        return data

class Helpers:
    @staticmethod
    def _collect_args(include_self:bool=False):
        """Utility: Collect all arguments of the caller as a dict."""
        frame = inspect.currentframe().f_back
        args, varargs, varkw, values = inspect.getargvalues(frame)
        if include_self:
            collected = {name: values[name] for name in args} # normal + defaults
        else:
            collected = {name: values[name] for name in args if name != "self"}
        if varargs:
            collected[varargs] = values[varargs]           # *args
        if varkw:
            collected.update(values[varkw])                # **kwargs
        return collected
    
    def _read_png_meta(self, png_path, restore=True):
        if hasattr(self, 'lock'):
            with self.lock:
                img = Image.open(png_path)
        else:
            img = Image.open(png_path)

        data = img.info.get('json_metadata')

        return (json.loads(data) if restore else data)

    @staticmethod
    def _invert_hex_color(hex_color: str, keep_hash: bool = True) -> str:
        """
        Invert an RGB hex color (opposite values: 255 - component).
        Accepts:
        - '#rgb' or 'rgb'
        - '#rrggbb' or 'rrggbb'
        - '#rgba' or 'rgba'
        - '#rrggbbaa' or 'rrggbbaa'
        Preserves alpha (if provided) and returns lowercase hex.
        Params:
        hex_color: input hex string
        keep_hash: whether to include leading '#' in return (default True)
        Returns:
        inverted hex string, e.g. '#00ff7f' or '#00ff7f80' (if alpha present)
        Raises:
        ValueError for invalid input lengths or invalid hex chars.
        """
        s = hex_color.strip()
        if s.startswith('#'):
            s = s[1:]
        s = s.lower()

        # Expand shorthand forms like 'abc' -> 'aabbcc', 'abcd' -> 'aabbccdd'
        if len(s) in (3, 4):
            s = ''.join(ch * 2 for ch in s)

        if len(s) not in (6, 8):
            raise ValueError(f"Invalid hex color length: {len(s)} for {hex_color!r}")

        # Validate hex characters
        try:
            int(s, 16)
        except ValueError:
            raise ValueError(f"Invalid hex color (non-hex digits): {hex_color!r}")

        rgb_part = s[:6]
        alpha_part = s[6:]  # '' if no alpha

        # parse components
        r = int(rgb_part[0:2], 16)
        g = int(rgb_part[2:4], 16)
        b = int(rgb_part[4:6], 16)

        # invert each channel
        r_inv = 255 - r
        g_inv = 255 - g
        b_inv = 255 - b

        inverted = f"{r_inv:02x}{g_inv:02x}{b_inv:02x}"
        if alpha_part:
            inverted += alpha_part  # preserve alpha exactly as given

        return ("#" if keep_hash else "") + inverted

    @staticmethod
    def _rgba_to_hex(rgba: Union[Sequence, object], keep_alpha: bool = False) -> str:
        """
        Accepts (`r,g,b`) or (`r,g,b,a`) where components are floats in `0..1` or ints in `0..255`.
        Returns `'#rrggbb'` by default. If `keep_alpha=True` and alpha provided, returns `'#rrggbbaa'`.
        """
        # handle numpy arrays, pandas Series, etc.
        if hasattr(rgba, "tolist"):
            rgba = rgba.tolist()

        if not isinstance(rgba, (list, tuple)):
            raise TypeError(f"Unsupported color type: {type(rgba)}")

        if len(rgba) < 3:
            raise ValueError(f"Color must have at least 3 components: {rgba!r}")

        def to_byte(v):
            if not isinstance(v, numbers.Real):
                # allow numeric strings? you could attempt float(v) here, but be explicit
                raise TypeError(f"Color component must be a number, got {type(v)} ({v!r})")
            # treat floats in [0,1] as normalized
            if isinstance(v, float) or (abs(v) <= 1 and not isinstance(v, int)):
                # Defensive: convert with float and test range
                fv = float(v)
                if 0.0 <= fv <= 1.0:
                    b = int(round(fv * 255))
                else:
                    # If float outside 0..1, treat as direct byte-ish and round
                    b = int(round(fv))
            else:
                b = int(round(v))
            # clamp to 0..255 to avoid bad hex formatting
            if b < 0:
                b = 0
            elif b > 255:
                b = 255
            return b

        r, g, b = rgba[0], rgba[1], rgba[2]
        r_x, g_x, b_x = map(to_byte, (r, g, b))

        if keep_alpha and len(rgba) >= 4:
            a = rgba[3]
            a_x = to_byte(a)
            return "#{:02x}{:02x}{:02x}{:02x}".format(r_x, g_x, b_x, a_x)
        return "#{:02x}{:02x}{:02x}".format(r_x, g_x, b_x)
    
    @staticmethod
    def _normalize_hex(s):
        """Normalize hex like '#abc' or 'abc' or '#aabbcc' -> '#aabbcc' (lowercase)."""
        s = s.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            raise ValueError(f"Invalid hex color: {s!r}")
        return "#" + s.lower()

    # ANSI color helpers for terminal output
    @staticmethod
    def _ansi_color(rgb):
        """Return ANSI escape code for background color from hex string."""
        if rgb.startswith('#'):
            rgb = rgb[1:]
        r, g, b = int(rgb[0:2],16), int(rgb[2:4],16), int(rgb[4:6],16)
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def _reset_color():
        return "\033[0m"
    
    def _new_seed(self):
        '''Generates a new random seed.'''
        seed=str(uuid.uuid4())
        if hasattr(self, 'print_on_new_seed') and self.print_on_new_seed:
            print('New seed: '+str(seed))
        return seed

    @staticmethod
    def _choose_black_or_white(
        bg_color: Union[str, Sequence],
        return_format: str = "hex"
    ) -> Union[str, Tuple[int,int,int]]:
        """
        Decide whether black or white text has better contrast on the given background color.

        Parameters:
        bg_color: hex string like '#123456' or 'fff', or a sequence (r,g,b) with ints (0..255)
                    or floats (0..1). Alpha is ignored if provided.
        return_format: 'hex' (default) -> '#000000' or '#ffffff'
                        'name' -> 'black' or 'white'
                        'rgb'  -> (r,g,b) tuple ints

        Uses WCAG relative luminance + contrast ratio to pick the color with higher contrast.
        """
        if return_format not in ("hex", "name", "rgb"):
            raise ValueError("return_format must be 'hex', 'name' or 'rgb'")
        
        def _relative_luminance(r: int, g: int, b: int) -> float:
            """
            Compute relative luminance (0..1) from integer RGB 0..255 using WCAG formula.
            """
            def _srgb_channel_to_linear(c: float) -> float:
                """
                Convert sRGB channel (0..1) to linear-light value per WCAG.
                c is in 0..1
                """
                if c <= 0.03928:
                    return c / 12.92
                return ((c + 0.055) / 1.055) ** 2.4
            rs = r / 255.0
            gs = g / 255.0
            bs = b / 255.0
            r_lin = _srgb_channel_to_linear(rs)
            g_lin = _srgb_channel_to_linear(gs)
            b_lin = _srgb_channel_to_linear(bs)
            return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

        
        def _parse_color_to_rgb255(color: Union[str, Sequence]) -> Tuple[int, int, int]:
            """Return (r,g,b) ints in 0..255 from hex string or sequence of numbers."""
            # hex string cases
            if isinstance(color, str):
                s = color.strip()
                if s.startswith('#'):
                    s = s[1:]
                s = s.lower()
                if len(s) in (3, 4):  # short form: rgb or rgba
                    s = ''.join(ch*2 for ch in s)
                if len(s) not in (6, 8):
                    raise ValueError(f"Invalid hex color length: {color!r}")
                try:
                    # parse rgb part, ignore alpha if present
                    r = int(s[0:2], 16)
                    g = int(s[2:4], 16)
                    b = int(s[4:6], 16)
                except ValueError:
                    raise ValueError(f"Invalid hex color: {color!r}")
                return r, g, b

            # sequence (list/tuple/numpy array-like)
            if hasattr(color, "tolist"):
                color = color.tolist()
            if isinstance(color, (list, tuple)):
                if len(color) < 3:
                    raise ValueError("Color sequence must have at least 3 components")
                comps = color[:3]
                out = []
                for v in comps:
                    if not isinstance(v, numbers.Real):
                        raise TypeError(f"Color component must be numeric, got {type(v)}")
                    # floats in 0..1 -> scale; ints assumed 0..255
                    if isinstance(v, float) or (abs(v) <= 1 and not isinstance(v, int)):
                        fv = float(v)
                        if 0.0 <= fv <= 1.0:
                            bv = int(round(fv * 255))
                        else:
                            bv = int(round(fv))
                    else:
                        bv = int(round(v))
                    # clamp
                    if bv < 0:
                        bv = 0
                    elif bv > 255:
                        bv = 255
                    out.append(bv)
                return tuple(out)

            raise TypeError(f"Unsupported color type: {type(color)}")

        r, g, b = _parse_color_to_rgb255(bg_color)
        L = _relative_luminance(r, g, b)

        # Contrast with black (L_black = 0) => (L + 0.05) / 0.05
        contrast_with_black = (L + 0.05) / 0.05
        # Contrast with white (L_white = 1) => (1 + 0.05) / (L + 0.05)
        contrast_with_white = (1.05) / (L + 0.05)

        use_white = contrast_with_white >= contrast_with_black

        if return_format == "hex":
            return "#ffffff" if use_white else "#000000"
        if return_format == "name":
            return "white" if use_white else "black"
        return (255, 255, 255) if use_white else (0, 0, 0)

class AddressHandler(Helpers):
    def __init__(
            self, 
            base_directory=os.getcwd(), 
            glyph_subfolder='glyphtables', 
            color_subfolder='colors',
            diacritics_subfolder='diacritics',
            print_on_new_seed:bool=True
        ):

        self.glyphs = Glyphs(base_directory, subfolder=glyph_subfolder)
        self.colors = Colors(base_directory, subfolder=color_subfolder)
        self.diacritics = Diacritics(base_directory, subfolder=diacritics_subfolder)
        
        self.lock = threading.RLock()
        self.print_on_new_seed = print_on_new_seed
    
    def table_from_seed(
            self,
            seed: Optional[str] = None,
            glyphs: str = 'Math1',
            colors: str = 'Beachgold',
            n: Optional[int] = None,
            d_weight_thresh: float = 0.5,
            d_left_of_string: bool = False
        ):

        # Validate inputs
        try:
            diacritics = self.diacritics.maps.get(drop_trailing_integers(glyphs))[0]
        except TypeError:
            diacritics = None
        glyphs = self.glyphs.maps.get(glyphs)[0] or self.glyphs.maps['Math1'][0]
        
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
            seed = self._new_seed()
        random.seed(seed)

        # Deterministic selection + padding by cycling if necessary
        import itertools

        # glyphs: pick up to n unique; if fewer, take all (shuffled) then cycle to n
        k_glyphs = min(n, len(glyphs))
        seed_glyphs = random.sample(glyphs, k_glyphs)
        if len(seed_glyphs) < n:
            seed_glyphs = list(itertools.islice(itertools.cycle(seed_glyphs), n))

        # Add diacritics
        if diacritics is not None:
            diacritic_weights = [random.random() for _ in range(n)]
            print(diacritic_weights)
            for index, weight in list(enumerate(diacritic_weights)):
                if d_weight_thresh > weight:
                    before = seed_glyphs[index]
                    if d_left_of_string:
                        seed_glyphs[index] = random.choice(diacritics) + seed_glyphs[index]
                    else:
                        seed_glyphs[index] = seed_glyphs[index] + random.choice(diacritics)
                    print(f'Grapheme "{seed_glyphs[index]}" from "{before}" ({d_weight_thresh}>{weight})')

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
        table = {
            i: {
                'glyph': seed_glyphs[i],
                'color': seed_colors[i]
            }
            for i in range(n)
        }
        return table
        
    def build(
            self, 
            seed:Optional[str]=None, 
            cols:int=9, 
            rows:int=1,
            glyphs:str='Math1', 
            colors:str='Beachgold',
            glyph_values:Optional[list[int|Any]]=None,
            color_values:Optional[list[int|Any]]=None,
            n:Optional[int]=None, # Impacts color gradient resolution
            d_weight_thresh:float=0.5,
            d_left_of_string: bool = False
        ):

        if glyph_values:
            if len(glyph_values) > cols*rows:
                glyph_values = glyph_values[:cols*rows]

        if color_values:
            if len(color_values) > cols*rows:
                color_values = color_values[:cols*rows]

        out = {}

        # Seed handling — ensure deterministic output
        if seed is None:
            seed = self._new_seed()

        table = self.table_from_seed(seed=seed, glyphs=glyphs, colors=colors, n=n, d_weight_thresh=d_weight_thresh, d_left_of_string=d_left_of_string)

        defaults = [random.choice(list(table.keys())) for _ in range(cols*rows)]

        for idx, dv in enumerate(defaults):
            out[idx] = {}

            # --- Glyph ---
            out[idx]['default_glyph'] = table[dv]['glyph']
            if glyph_values and idx < len(glyph_values):
                gv = glyph_values[idx]
                if isinstance(gv, int):
                    # Wrap around using modulo
                    key = list(table.keys())[gv % len(table)]
                else:
                    # If string or something else, fallback to default
                    key = gv if gv in table else dv
            else:
                key = dv
            out[idx]['glyph'] = table[key]['glyph']

            # --- Color ---
            out[idx]['default_color'] = table[dv]['color']
            if color_values and idx < len(color_values):
                cv = color_values[idx]
                if isinstance(cv, int):
                    key = list(table.keys())[cv % len(table)]
                else:
                    key = cv if cv in table else dv
            else:
                key = dv
            out[idx]['color'] = table[key]['color']

        return out
    
    # TODO : Markdown output
    # TODO : HTML output
    def __call__(
            self, 
            
            seed:Optional[str]=None, 
            seed_prefix:str='PTR-',
            seed_length:Optional[int]=None,
            
            cols:int=9, 
            rows:int=1,
            
            glyphs:str='Math1', 
            colors:str='Beachgold',
            
            glyph_values:Optional[list[int|Any]]=None,
            color_values:Optional[list[int|Any]]=None,
            n:Optional[int]=None, # Impacts color gradient resolution
            d_weight_thresh:float = 0.5,
            d_left_of_string: bool = False,
            
            mode:Literal['terminal', 'png', 'both', 'none']='terminal',
            png_path:Optional[str]=None,
            font_size:int=26,
            font_colors:Literal['black', 'white', 'auto', 'inverted']='auto',
            
            small_default:bool=False,
            small_size:int=12,
            small_color:Optional[str]=None, # None = inverted

            dpi:int=300,
            lower_or_upper:Optional[Literal['lower', 'upper']]=None,

            embed_json_data:bool=True,

            *args, **kwargs
        ):
        """
        Generate a deterministic grid of glyphs/colors and optionally render to terminal and/or PNG.

        The generation is deterministic when `seed` is provided; otherwise a UUID seed is created
        and printed. Uses `table_from_seed()` to build a lookup table of glyph/color pairs.

        :param seed: Deterministic seed string (UUID-like). If ``None``, a new UUID is generated.
        :type seed: Optional[str]
        :param seed_prefix: Prefix added to the printed/output identifier (default ``'PTR-'``).
        :type seed_prefix: str
        :param seed_length: If provided, truncate the seed to this length for the output id.
        :type seed_length: Optional[int]
        :param cols: Number of columns in the grid.
        :type cols: int
        :param rows: Number of rows in the grid.
        :type rows: int
        :param glyphs: Key in ``Glyphs.maps`` selecting the glyph set (falls back to ``'Math1'``).
        :type glyphs: str
        :param colors: Key in ``Colors.maps`` selecting the colormap (falls back to ``'Beachgold'``).
        :type colors: str
        :param glyph_values: Optional per-cell glyph selectors. Integers use modulo indexing; other
                            values are treated as keys when present in the table.
        :type glyph_values: Optional[list]
        :param color_values: Optional per-cell color selectors; semantics like ``glyph_values``.
        :type color_values: Optional[list]
        :param n: Controls color gradient resolution (passed to ``Colors.colormap``). When ``None``
                it uses the colormap length.
        :type n: Optional[int]
        :param mode: Output mode: ``'terminal'``, ``'png'``, ``'both'`` or ``'none'``.
        :type mode: Literal['terminal','png','both','none']
        :param png_path: Path to save PNG (defaults to ``seed + ".png"`` when PNG mode used).
        :type png_path: Optional[str]
        :param font_size: Font size for main glyphs; if ``None`` uses glyph table configured size.
        :type font_size: int
        :param font_colors: Foreground selection: ``'black'``, ``'white'``, ``'auto'`` (WCAG), or
                            ``'inverted'`` (invert bg color).
        :type font_colors: Literal['black','white','auto','inverted']
        :param small_default: If True, render the default glyph in the corner when different.
        :type small_default: bool
        :param small_size: Font size for the small default glyph (defaults to ~half main size).
        :type small_size: int
        :param small_color: Hex color for the small glyph; when ``None`` it uses inverted cell color.
        :type small_color: Optional[str]
        :param dpi: DPI used when rendering PNG.
        :type dpi: int
        :param lower_or_upper: Force output identifier case; ``'lower'`` or ``'upper'``.
        :type lower_or_upper: Optional[Literal['lower','upper']]
        :param args: Forward-compatible positional args (unused).
        :param kwargs: Forward-compatible keyword args (unused).

        :returns: If ``mode == 'none'`` returns the generated table (dict mapping index -> cell dict).
                Otherwise returns the PNG path string (``png_path``).
        :rtype: Union[dict, str]

        :raises ValueError: If no glyphs or no colors are available, or invalid color/hex lengths.
        :raises TypeError: If malformed types are provided for colors/glyphs.
        :raises OSError: If font files are missing or matplotlib raises on font load/save.

        :note: PNG saving is protected by an RLock (`self.lock`) to avoid concurrent writes.
        :note: ``font_colors='auto'`` uses WCAG relative-luminance to choose black or white.
        """

        _args = self._collect_args()

        # Validate inputs

        def to_int(v):
            try:
                return int(v)
            except Exception:
                return None

        def safe_value_convert(vals):
            if isinstance(vals, str):
                vals = vals.split(',')
                vals = [to_int(v) for v in vals]
                return vals
            else:
                return vals
        
        glyph_values = safe_value_convert(glyph_values)
        color_values = safe_value_convert(color_values)

        seed = seed or self._new_seed()
        glyphs = glyphs if glyphs in self.glyphs.maps else 'Math1'
        colors = colors if colors in self.colors.maps else 'Beachgold'

        if font_size is None:
            font_size = self.glyphs.maps.get(glyphs)[2] or 26
        small_size = small_size or int(round(font_size / 2))
        font_path = os.path.join(self.glyphs.fontdir, self.glyphs.maps.get(glyphs)[1])

        output_ident = seed if seed_length is None else seed[:seed_length].replace('-','')
        output_ident = output_ident.lower() if lower_or_upper == 'lower' else output_ident.upper() if lower_or_upper == 'upper' else output_ident
        if seed_prefix:
            output_ident = f"{seed_prefix}{output_ident}"

        # Obtain build table

        table = self.build(
            seed=seed, 
            cols=cols, 
            rows=rows,
            glyphs=glyphs, 
            colors=colors,
            glyph_values=glyph_values,
            color_values=color_values,
            n=n,
            d_weight_thresh=d_weight_thresh,
            d_left_of_string=d_left_of_string
        )

        if mode in ['none', None]:
            return table

        if mode in ['png', 'both']:
            png_path = png_path or seed+".png" # Patch png_path
            fig, ax = plt.subplots(figsize=(cols, rows+1), dpi=dpi)
            fig.patch.set_alpha(0)
            ax.set_xlim(0, cols)
            ax.set_ylim(0, rows+1)
            ax.axis('off')
            ax.set_facecolor('none')
            font_properties = fm.FontProperties(fname=font_path) if font_path else None

            for i in range(rows):
                for j in range(cols):
                    symbol = table[i*cols + j]['glyph']
                    color = table[i*cols + j]['color'] # the bg color
                    symbol_color = (
                        self._invert_hex_color(color) if font_colors == 'inverted'
                        else self._choose_black_or_white(color) if font_colors == 'auto'
                        else font_colors
                    )
                    
                    # small_default
                    default_symbol = table[i*cols + j]['default_glyph']
                    #default_color = table[i*cols + j]['default_color']
                    _small_color = small_color or self._invert_hex_color(color)
                    
                    rect = mpatches.Rectangle(
                        (j, rows-i-0.5), 
                        1, 
                        1, 
                        color=color
                    )
                    ax.add_patch(rect)
                    ax.text(
                        j+0.5, 
                        rows-i, 
                        symbol, 
                        ha='center', 
                        va='center', 
                        color=symbol_color, 
                        fontsize=font_size, 
                        fontproperties=font_properties
                    )

                    # put small default symbol
                    if default_symbol != symbol and small_default:
                        ax.text(
                            j+0.80, 
                            rows-i-0.33, 
                            default_symbol, 
                            ha='center', 
                            va='center', 
                            color=_small_color, 
                            fontsize=small_size, 
                            fontproperties=font_properties,
                            alpha=0.7
                        )

            ax.text(
                cols/2, 
                0.15, 
                output_ident,
                ha='center', 
                va='center', 
                fontsize=16, 
                color='gray'
            )
            
            with self.lock:
                plt.savefig(png_path, bbox_inches='tight', dpi=dpi, transparent=True)
            
            # Encode the arguments
            if embed_json_data:
                with self.lock:
                    img=Image.open(png_path)
                meta = PngImagePlugin.PngInfo()
                meta.add_text('json_metadata', json.dumps(_args))
                with self.lock:
                    img.save(png_path, pnginfo=meta)

            plt.close(fig)

        # Get terminal output

        if mode in ['terminal', 'both']:
            lines = []
            current_row = []

            for idx, gt in enumerate(table.values()):
                cell = self._ansi_color(gt['color']) + f" {gt['glyph']} " + self._reset_color()
                current_row.append(cell)

                # End of row
                if (idx + 1) % cols == 0:
                    lines.append("".join(current_row))
                    current_row = []

            # If the last row isn’t full (edge case)
            if current_row:
                lines.append("".join(current_row))

            for line in lines:
                print(line)

        return png_path