"""Render real command output (stdin) to a terminal-style PNG. Usage: cmd | python scripts/render_terminal.py out.png "title" """
import sys

from PIL import Image, ImageDraw, ImageFont

lines = sys.stdin.read().rstrip().splitlines()
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 18)
w = max(900, 20 + max(int(font.getlength(l)) for l in lines) + 20)
img = Image.new("RGB", (w, 60 + 24 * len(lines)), "#1e1e2e")
d = ImageDraw.Draw(img)
d.text((20, 15), sys.argv[2], fill="#89b4fa", font=font)
for i, l in enumerate(lines):
    col = "#a6e3a1" if ("passed" in l or l.startswith("A:")) else "#cdd6f4"
    d.text((20, 50 + 24 * i), l, fill=col, font=font)
img.save(sys.argv[1])
