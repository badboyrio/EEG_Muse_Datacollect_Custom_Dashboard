"""Re-render the dashboard demo animation (docs/demo/dashboard_demo.gif / .mp4).

dashboard_demo.html is a static recreation of the dashboard UI driven by simulated EEG.
Its render(t) function draws the state at t seconds; this script screenshots it at 10 fps.

    pip install playwright && playwright install chromium
    python docs/demo/render_demo.py            # frames -> docs/demo/frames/
    # then encode (needs ffmpeg):
    ffmpeg -framerate 10 -i docs/demo/frames/f%04d.png -vf "fps=8,scale=960:-1:flags=lanczos,palettegen=max_colors=160" pal.png
    ffmpeg -framerate 10 -i docs/demo/frames/f%04d.png -i pal.png -lavfi "fps=8,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle" -loop 0 docs/demo/dashboard_demo.gif
    ffmpeg -framerate 10 -i docs/demo/frames/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 docs/demo/dashboard_demo.mp4
"""
import os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frames")
SECONDS, FPS = 24, 10

os.makedirs(OUT, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1262})
    page.goto("file://" + os.path.join(HERE, "dashboard_demo.html"))
    page.wait_for_timeout(300)
    for i in range(SECONDS * FPS):
        page.evaluate(f"render({i / FPS})")
        page.screenshot(path=os.path.join(OUT, f"f{i:04d}.png"))
    browser.close()
print(f"{SECONDS * FPS} frames in {OUT}")
