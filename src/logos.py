"""Local disk cache for team logo images, used as scatter-plot markers."""

from pathlib import Path
from urllib.request import urlopen

from PIL import Image

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "logos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_logo(team: str, url: str):
    path = CACHE_DIR / f"{team}.png"
    if not path.exists():
        try:
            with urlopen(url, timeout=10) as resp:
                path.write_bytes(resp.read())
        except Exception:
            return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None
