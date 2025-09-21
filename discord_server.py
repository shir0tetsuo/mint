# discord_server.py
import asyncio
import functools
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence
import uuid

import discord
from discord import app_commands
from discord.ext import commands

# Import your existing generation module
import to_terminal

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("glyph-bot")

# ---------- Bot setup ----------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
TREE = bot.tree

# ---------- Config ----------
ROOT_DIR = Path(__file__).parent
PAGE_SIZE = 25  # Discord select menu max options per select

# ---------- Helpers ----------
def _list_files_folder(folder: Path) -> Sequence[str]:
    if not folder.exists() or not folder.is_dir():
        return []
    items = []
    for p in sorted(folder.iterdir()):
        if p.is_file():
            items.append(p.stem)
        elif p.is_dir():
            items.append(p.name)
    return items

def _get_glyphtable_options() -> List[str]:
    fs_opts = list(_list_files_folder(ROOT_DIR / "glyphtables"))
    if fs_opts:
        log.info("Loaded %d glyphtables from ./glyphtables", len(fs_opts))
        return fs_opts
    fallback = list(getattr(to_terminal, "GLYPH_TABLES",
                            getattr(to_terminal, "glyphtable_options",
                                    getattr(to_terminal, "GLYPHTABLE_OPTIONS", ["default_glyphs"]))))
    log.info("Using fallback glyphtable list of length %d", len(fallback))
    return fallback

def _get_color_options() -> List[str]:
    fs_opts = list(_list_files_folder(ROOT_DIR / "colors"))
    if fs_opts:
        log.info("Loaded %d colors from ./colors", len(fs_opts))
        return fs_opts
    fallback = list(getattr(to_terminal, "COLOR_MAPS",
                            getattr(to_terminal, "color_options",
                                    getattr(to_terminal, "CMAP_OPTIONS", ["viridis", "gray"]))))
    log.info("Using fallback color list of length %d", len(fallback))
    return fallback

GLYPH_ALL = _get_glyphtable_options()
COLOR_ALL = _get_color_options()

# ---------- Interaction debug ----------
@bot.event
async def on_interaction(interaction: discord.Interaction):
    try:
        cmd_name = getattr(interaction, "command_name", None)
        guild_id = getattr(interaction.guild, "id", None) if interaction.guild else None
        print("DEBUG on_interaction:",
              "type=", interaction.type,
              "command=", cmd_name,
              "user=", f"{interaction.user}#{getattr(interaction.user, 'discriminator', '')}",
              "user_id=", getattr(interaction.user, "id", None),
              "guild_id=", guild_id)
    except Exception:
        log.exception("Error in on_interaction debug hook")

# ---------- UI: paginated select + paging buttons ----------
def _build_options_page(items: Sequence[str], page: int, page_size: int = PAGE_SIZE):
    start = page * page_size
    page_items = items[start:start + page_size]
    return [discord.SelectOption(label=i, value=i) for i in page_items], start, len(items)

class PaginatedStringSelect(discord.ui.Select):
    """
    A Select whose options are built from a page of a larger list. The parent view
    should set .all_items and .page on creation; callback stores selection on view.
    """
    def __init__(self, name: str, all_items: Sequence[str], page: int = 0):
        self.name = name
        self.all_items = list(all_items)
        self.page = max(0, int(page))
        options, start, total = _build_options_page(self.all_items, self.page)
        placeholder = f"Select {name} (page {self.page+1} of {max(1, (total-1)//PAGE_SIZE + 1)})"
        super().__init__(
            placeholder=placeholder,
            options=options,
            min_values=1,
            max_values=1,
            custom_id=f"{name}_select_{self.page}"
        )

    def rebuild(self):
        """Recompute options & placeholder after changing self.page"""
        options, start, total = _build_options_page(self.all_items, self.page)
        self.options = options
        self.custom_id = f"{self.name}_select_{self.page}"
        self.placeholder = f"Select {self.name} (page {self.page+1} of {max(1, (total-1)//PAGE_SIZE + 1)})"

    async def callback(self, interaction: discord.Interaction):
        view: AddressView = self.view  # type: ignore
        chosen = self.values[0]
        if self.name == "glyphtable":
            view.glyphtable = chosen
        elif self.name == "color":
            view.color = chosen
        else:
            setattr(view, self.name, chosen)
        await view.maybe_generate(interaction)

class PageButton(discord.ui.Button):
    def __init__(self, label: str, target: str, delta: int):
        """
        label: shown label (Prev/Next)
        target: 'glyphtable' or 'color'
        delta: -1 or +1
        """
        super().__init__(style=discord.ButtonStyle.secondary, label=label)
        self.target = target
        self.delta = int(delta)

    async def callback(self, interaction: discord.Interaction):
        view: AddressView = self.view  # type: ignore
        select_obj = None
        for item in view.children:
            if isinstance(item, PaginatedStringSelect) and item.name == self.target:
                select_obj = item
                break
        if select_obj is None:
            try:
                await interaction.response.send_message("Nothing to page.", ephemeral=True)
            except Exception:
                pass
            return

        total_pages = max(1, (len(select_obj.all_items) - 1) // PAGE_SIZE + 1)
        new_page = max(0, min(total_pages - 1, select_obj.page + self.delta))
        if new_page == select_obj.page:
            try:
                await interaction.response.defer(ephemeral=True, thinking=False)
            except Exception:
                pass
            return

        select_obj.page = new_page
        select_obj.rebuild()

        try:
            await interaction.response.edit_message(view=view)
        except Exception:
            try:
                await interaction.followup.send("Updated page.", ephemeral=True)
            except Exception:
                pass

# ---------- The View ----------
class AddressView(discord.ui.View):
    def __init__(self, author: discord.User, params: dict, timeout: float = 120.0):
        super().__init__(timeout=timeout)
        self.author = author
        self.params = params or {}
        self.glyphtable: Optional[str] = None
        self.color: Optional[str] = None

        # Add glyphtable select + paging buttons (if any items)
        if GLYPH_ALL:
            gly_select = PaginatedStringSelect("glyphtable", GLYPH_ALL, page=0)
            self.add_item(gly_select)
            self.add_item(PageButton("Prev Glyphs", target="glyphtable", delta=-1))
            self.add_item(PageButton("Next Glyphs", target="glyphtable", delta=+1))
        else:
            log.warning("No glyphtable options available; ./glyphtables is empty or fallback empty.")

        # Add color select + paging buttons (if any items)
        if COLOR_ALL:
            color_select = PaginatedStringSelect("color", COLOR_ALL, page=0)
            self.add_item(color_select)
            self.add_item(PageButton("Prev Colors", target="color", delta=-1))
            self.add_item(PageButton("Next Colors", target="color", delta=+1))
        else:
            log.warning("No color options available; ./colors is empty or fallback empty.")

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        is_allowed = interaction.user.id == self.author.id
        if not is_allowed:
            try:
                await interaction.response.send_message("You may not interact with this control.", ephemeral=True)
            except Exception:
                pass
        return is_allowed

    async def maybe_generate(self, interaction: discord.Interaction) -> None:
        need_glyph = bool(GLYPH_ALL)
        need_color = bool(COLOR_ALL)

        if (need_glyph and not self.glyphtable) or (need_color and not self.color):
            try:
                await interaction.response.defer(ephemeral=True, thinking=False)
            except Exception:
                pass
            return

        try:
            await interaction.response.defer(thinking=True)
        except Exception:
            pass

        # Prepare generator kwargs, with defensive casting where appropriate
        gen_kwargs = {
            "glyphtable": self.glyphtable,
            "cmap": self.color,
            "seed": self.params.get("seed"),
            "rows": int(self.params.get("rows")) if self.params.get("rows") is not None else None,
            "cols": int(self.params.get("cols")) if self.params.get("cols") is not None else None,
            "passed_uuid": self.params.get("passed_uuid"),       # note: avoid 'uuid' name
            "shorten_uuid": int(self.params.get("shorten_uuid")) if self.params.get("shorten_uuid") is not None else None,
            "fsize": self.params.get("fsize"),
            "glyph_values": self.params.get("glyph_values"),
            "color_values": self.params.get("color_values"),
        }

        # prune None values to avoid shadowing modules or passing accidental None
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        loop = asyncio.get_running_loop()
        out_path = None
        try:
            gen_func = functools.partial(to_terminal.generate_glyph_png, **gen_kwargs)
            out_path = await loop.run_in_executor(None, gen_func)

            if out_path is None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.close()
                gen_kwargs_with_out = dict(gen_kwargs)
                gen_kwargs_with_out["out_path"] = tmp.name
                out_path = await loop.run_in_executor(None, functools.partial(to_terminal.generate_glyph_png, **gen_kwargs_with_out))

                if out_path is None:
                    raise RuntimeError("Generator did not return an output path.")
        except Exception as exc:
            log.exception("Generation failed")
            try:
                await interaction.followup.send(f"Failed to generate image: {exc}", ephemeral=True)
            except Exception:
                pass
            return

        try: #.\n||`{gen_kwargs}`||
            print(gen_kwargs)
            await interaction.followup.send(content=f"Generated with glyphtable `{self.glyphtable}` and color `{self.color}`", file=discord.File(out_path, filename=os.path.basename(out_path)))
        except Exception as exc:
            log.exception("Failed to send followup")
            try:
                await interaction.followup.send(f"Failed to send image: {exc}", ephemeral=True)
            except Exception:
                pass
        finally:
            try:
                if out_path and os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                log.warning("Could not remove temp file %s", out_path)
            self.stop()

# ---------- Slash command (application command) ----------
@TREE.command(name="uuid", description="Generate a random UUID")
async def uuid_command(interaction: discord.Interaction):
    random_uuid = str(uuid.uuid4())
    try:
        await interaction.response.send_message(f"`{random_uuid}`", ephemeral=True)
    except Exception:
        log.exception("Failed to send UUID message")
        try:
            await interaction.response.send_message("Failed to generate UUID; see bot logs.", ephemeral=True)
        except Exception:
            pass



@TREE.command(name="address", description="Open the glyph generation UI")
@app_commands.describe(
    seed="Optional seed string for deterministic generation",
    rows="Number of rows in the glyph address (default 2)",
    cols="Number of columns in the glyph address (default 8)",
    uuid="Optional UUID string to use instead of generating a new one",
    shorten_uuid="If set, shortens the UUID to this many characters",
    fsize="Font size for rendering (optional)",
    glyph_values="Custom glyph values (optional)",
    color_values="Custom color values (optional)"
)
async def address(interaction: discord.Interaction,
                  seed: Optional[str] = None,
                  rows: Optional[int] = 2,
                  cols: Optional[int] = 8,
                  uuid: Optional[str] = None,
                  shorten_uuid: Optional[int] = None,   # <-- changed to int per your note
                  fsize: Optional[int] = None,
                  glyph_values: Optional[str] = None,
                  color_values: Optional[str] = None):
    # ensure we have at least one option available per select (or else inform the user)
    missing = []
    if not GLYPH_ALL:
        missing.append("./glyphtables (no files found)")
    if not COLOR_ALL:
        missing.append("./colors (no files found)")
    if missing:
        await interaction.response.send_message(
            f"Cannot open glyph UI — missing options: {', '.join(missing)}. Please add files to those folders.",
            ephemeral=True
        )
        return

    # Map the incoming 'uuid' param to 'passed_uuid' used by generator to avoid shadowing uuid module
    params = {
        "seed": seed,
        "rows": rows,
        "cols": cols,
        "passed_uuid": uuid,
        "shorten_uuid": shorten_uuid,
        "fsize": fsize,
        "glyph_values": glyph_values,
        "color_values": color_values,
    }

    view = AddressView(interaction.user, params, timeout=180.0)

    try:
        await interaction.response.send_message("Pick a glyphtable and a color map (use Prev/Next to page):", view=view, ephemeral=True)
    except Exception:
        log.exception("Failed to send address UI message")
        try:
            await interaction.response.send_message("Failed to open glyph UI; see bot logs.", ephemeral=True)
        except Exception:
            pass

# ---------- Tree sync and ready ----------
@bot.event
async def on_ready():
    log.info("Bot ready. Logged in as %s (%s)", bot.user, getattr(bot.user, "id", None))
    try:
        synced = await TREE.sync()
        log.info("Application commands synced. Count: %s", len(synced))
    except Exception:
        log.exception("Failed to sync application commands")

# ---------- Run ----------
if __name__ == "__main__":
    TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "<PUT_YOUR_TOKEN_HERE>")
    if TOKEN == "<PUT_YOUR_TOKEN_HERE>":
        log.error("You must set DISCORD_BOT_TOKEN environment variable or update the script.")
    else:
        bot.run(TOKEN)
