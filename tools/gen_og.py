"""Generate social share cards (and the favicon) for residual-thoughts.com.

Usage, from the repo root:

    python tools/gen_og.py              # create og.png for any post missing one
    python tools/gen_og.py --post SLUG  # (re)generate one post's card
    python tools/gen_og.py --force      # regenerate everything, incl. favicon
                                        # and og_default.png

For a new post, run this once and add to its front matter either
`image: figs/og.png` (posts without figures) or, to keep a figure as the
listing thumbnail:

    open-graph:
      image: figs/og.png
    twitter-card:
      image: figs/og.png
      card-style: summary_large_image

Palette and motif match styles.css: teal read (#2E7D74) -> violet write
(#6B5CA5) gradient spine on paper (#FBFAF6), ink text (#1C1F2E).
"""
import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent

PAPER = (251, 250, 246, 255)
INK = (28, 31, 46, 255)
PENCIL = (139, 136, 124, 255)
HAIRLINE = (230, 226, 214, 255)
READ = (46, 125, 116, 255)
WRITE = (107, 92, 165, 255)

FONTS = {
    "serif_bold": r"C:\Windows\Fonts\georgiab.ttf",
    "mono": r"C:\Windows\Fonts\consola.ttf",
    "mono_bold": r"C:\Windows\Fonts\consolab.ttf",
}

SITE_NAME = "RESIDUAL THOUGHTS"
SITE_URL = "residual-thoughts.com"
SITE_TAGLINE = "Notes on mechanistic interpretability, geometry, and model behavior."
DEFAULT_TAGS = ["residual streams", "attention heads", "sae features"]


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(4))


def draw_vgrad_bar(draw, x0, y0, x1, y1, c_top, c_bot):
    h = y1 - y0
    for y in range(y0, y1):
        t = (y - y0) / max(h - 1, 1)
        draw.line([(x0, y), (x1 - 1, y)], fill=lerp(c_top, c_bot, t))


def spaced_text(draw, xy, text, font, fill, tracking=0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def wrap_title(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def og_card(path, title, tags):
    W, H = 1200, 630
    img = Image.new("RGBA", (W, H), PAPER)
    d = ImageDraw.Draw(img)

    # spine with read tap + write node at title height
    draw_vgrad_bar(d, 0, 0, 16, H, READ, WRITE)
    tap_y = 208
    d.line([(16, tap_y), (66, tap_y)], fill=READ, width=4)
    d.ellipse([66 - 9, tap_y - 9, 66 + 9, tap_y + 9], fill=WRITE)

    f_eye = ImageFont.truetype(FONTS["mono_bold"], 27)
    spaced_text(d, (92, 76), SITE_NAME, f_eye, PENCIL, tracking=6)

    # title, shrink-to-fit
    max_w = 1010
    size = 66
    while True:
        f_t = ImageFont.truetype(FONTS["serif_bold"], size)
        lines = wrap_title(d, title, f_t, max_w)
        line_h = int(size * 1.22)
        if line_h * len(lines) <= 300 or size <= 46:
            break
        size -= 4
    y = tap_y - int(size * 0.52)
    for ln in lines:
        d.text((92, y), ln, font=f_t, fill=INK)
        y += line_h

    # tag chips
    f_tag = ImageFont.truetype(FONTS["mono"], 23)
    cx, cy = 92, y + 26
    for tag in tags[:3]:
        tw = d.textlength(tag, font=f_tag)
        if cx + tw + 30 > W - 92:
            break
        d.rounded_rectangle([cx, cy, cx + tw + 30, cy + 42], radius=6,
                            outline=HAIRLINE, width=2)
        d.text((cx + 15, cy + 8), tag, font=f_tag, fill=READ)
        cx += tw + 30 + 14

    f_foot = ImageFont.truetype(FONTS["mono"], 24)
    d.text((92, 556), SITE_URL, font=f_foot, fill=PENCIL)

    img.convert("RGB").save(path, "PNG")
    print("wrote", path.relative_to(REPO))


def favicon(path):
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bx0, bx1, by0, by1 = 112, 146, 18, 238
    draw_vgrad_bar(d, bx0, by0, bx1, by1, READ, WRITE)
    r = (bx1 - bx0) // 2
    d.ellipse([bx0, by0 - r, bx1 - 1, by0 + r], fill=READ)
    d.ellipse([bx0, by1 - r, bx1 - 1, by1 + r], fill=WRITE)
    cy = 84
    d.ellipse([129 - 44, cy - 44, 129 + 44, cy + 44], fill=WRITE)
    img.save(path, "PNG")
    print("wrote", path.relative_to(REPO))


def post_meta(qmd):
    text = qmd.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    front = text.split("---", 2)[1]
    meta = yaml.safe_load(front)
    title = meta.get("title")
    cats = [c for c in meta.get("categories", [])
            if c != "mechanistic-interpretability"] or meta.get("categories", [])
    return title, cats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--post", help="post slug (directory name under posts/)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the file exists")
    args = ap.parse_args()

    posts = sorted(p for p in (REPO / "posts").iterdir()
                   if (p / "index.qmd").exists())
    if args.post:
        posts = [p for p in posts if p.name == args.post]
        if not posts:
            sys.exit(f"no post named {args.post!r} under posts/")

    for post in posts:
        out = post / "figs" / "og.png"
        if out.exists() and not (args.force or args.post):
            print("skip (exists)", out.relative_to(REPO))
            continue
        meta = post_meta(post / "index.qmd")
        if not meta or not meta[0]:
            print("skip (no title)", post.name)
            continue
        out.parent.mkdir(exist_ok=True)
        og_card(out, meta[0], meta[1])

    if not args.post:
        for name, gen, gen_args in [
            ("og_default.png", og_card, (SITE_TAGLINE, DEFAULT_TAGS)),
            ("favicon.png", favicon, ()),
        ]:
            target = REPO / name
            if target.exists() and not args.force:
                print("skip (exists)", name)
            else:
                gen(target, *gen_args)


if __name__ == "__main__":
    main()
