"""Record the studio being used, for the demo: pick a shot, cue it, make the shot, re-time it.

    python tools/demo/record_studio.py --plate "Toy dinosaur, night" --say samples/voices/fwoooom.wav -o runs/demo/studio
    python tools/demo/record_studio.py --plate "Candle jar" --sample "pf." -o runs/demo/studio_offline

--say: the browser's microphone plays that file (Chromium's fake capture device), so the take goes through the
same microphone path a person's voice does; --sample picks a sample noise; --clip uses the clip's own sound.
Needs Playwright (pip install playwright) and a Chromium (--chromium, else Playwright's own). Writes studio.webm
with a visible cursor and marks.json (seconds into the video of each step), so the edit can cut the render wait.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

CURSOR = """
window.addEventListener('DOMContentLoaded', () => {
  const hide = document.createElement('style');  // the framework's dev-mode badge
  hide.textContent = 'nextjs-portal { display: none !important; }';
  document.head.appendChild(hide);
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;'
    + 'background:rgba(255,255,255,.85);border:2px solid #111;z-index:2147483647;pointer-events:none;'
    + 'transition:transform .12s;box-shadow:0 0 8px rgba(0,0,0,.5)';
  document.body.appendChild(c);
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', () => { c.style.transform = 'scale(.7)'; }, true);
  addEventListener('mouseup', () => { c.style.transform = 'scale(1)'; }, true);
});
"""


class Recorder:
    def __init__(self, page: Page):
        self.page, self.t0, self.marks = page, time.perf_counter(), {}

    def mark(self, name: str) -> None:
        self.marks[name] = round(time.perf_counter() - self.t0, 2)

    def pause(self, s: float) -> None:
        self.page.wait_for_timeout(int(s * 1000))

    def click(self, locator, pause: float = 0.6) -> None:
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.mouse.move(x, y, steps=18)
        self.pause(0.25)
        self.page.mouse.click(x, y)
        self.pause(pause)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:3000")
    ap.add_argument("--plate", required=True, help="the sample shot's title, as the studio shows it")
    cue = ap.add_mutually_exclusive_group(required=True)
    cue.add_argument("--say", help="a WAV the fake microphone plays while you perform")
    cue.add_argument("--sample", help="a sample noise's title")
    cue.add_argument("--clip", action="store_true", help="lock it to the clip's own sound")
    ap.add_argument("--draw", help='draw instead of tapping: one stroke as JSON points on the plate, "[[x, y], ...]"')
    ap.add_argument("--brush", default="medium", help="the brush to draw with, as the studio labels it")
    ap.add_argument("--describe", help="type this into 'Say what happens'")
    ap.add_argument("--retime", type=float, default=0.03, help="drag the hit by this share of the ribbon (0: don't)")
    ap.add_argument("--chromium", help="a Chromium executable (default: Playwright's)")
    ap.add_argument("-o", "--out", default="runs/demo/studio")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = ["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
    if a.say:
        args += ["--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={Path(a.say).resolve()}%noloop"]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chromium, args=args)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1,
                                  record_video_dir=str(out / "raw"), record_video_size={"width": 1280, "height": 800},
                                  permissions=["microphone"])
        ctx.add_init_script(CURSOR)
        page = ctx.new_page()
        r = Recorder(page)
        page.goto(a.url)
        page.get_by_text(a.plate, exact=True).first.wait_for(timeout=30000)
        r.pause(1.5)
        r.mark("pick")
        r.click(page.get_by_text(a.plate, exact=True).first, pause=1.2)
        r.click(page.get_by_role("button", name="Move it"), pause=1.5)  # the spot comes with the sample: show it
        if a.draw:
            r.click(page.get_by_role("radio", name="Draw it"), pause=1.0)
            r.click(page.get_by_role("radio", name=a.brush), pause=0.6)
            pad = page.locator('canvas[aria-label="your drawing"]')
            pad.scroll_into_view_if_needed()
            box = pad.bounding_box()
            pts = [(box["x"] + x * box["width"], box["y"] + y * box["height"]) for x, y in json.loads(a.draw)]
            page.mouse.move(*pts[0], steps=18)
            r.pause(0.3)
            r.mark("draw")
            page.mouse.down()
            for p_ in pts[1:]:
                page.mouse.move(*p_, steps=10)
            page.mouse.up()
            r.pause(1.2)
            r.click(page.get_by_role("button", name="That's where it goes"), pause=1.0)
        else:
            r.click(page.get_by_role("button", name="That's the spot"), pause=1.0)
        r.mark("cue")
        if a.say:
            r.click(page.get_by_role("button", name="Use my microphone"), pause=1.5)
            r.click(page.get_by_role("button", name="Perform"), pause=0.5)
            r.mark("performing")
            page.get_by_role("heading", name="4. Make the shot").wait_for(timeout=60000)
        elif a.clip:
            r.click(page.get_by_role("radio", name="The clip's own sound"), pause=1.0)
            r.click(page.get_by_role("button", name="Lock it to my clip's sound"), pause=1.0)
        else:
            r.click(page.locator("button", has_text=a.sample).first, pause=1.0)
        if a.describe:
            field = page.get_by_label("Say what happens")
            r.click(field, pause=0.3)
            page.keyboard.type(a.describe, delay=35)
        r.pause(1.5)
        r.mark("make")
        r.click(page.get_by_role("button", name="Make the shot"), pause=0.5)
        page.locator("video.result").wait_for(timeout=900000)
        r.mark("done")
        page.locator("video.result").scroll_into_view_if_needed()
        r.pause(7.0)
        if a.retime:
            ribbon = page.locator("svg.ribbon")
            ribbon.scroll_into_view_if_needed()
            handle = ribbon.locator("g").first.locator("rect").first
            hb, rb = handle.bounding_box(), ribbon.bounding_box()
            x, y = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
            page.mouse.move(x, y, steps=18)
            r.pause(0.4)
            r.mark("drag")
            page.mouse.down()
            page.mouse.move(x + a.retime * rb["width"], y, steps=24)
            page.mouse.up()
            r.pause(1.0)
            r.click(page.get_by_role("button", name="Re-time it"), pause=0.5)
            page.get_by_role("button", name="Re-time it").wait_for(timeout=120000)
            page.wait_for_function("() => ![...document.querySelectorAll('button')].some(b => b.textContent.includes('…'))",
                                   timeout=120000)
            r.mark("retimed")
            page.locator("video.result").scroll_into_view_if_needed()
            r.pause(6.0)
        r.mark("end")
        video = page.video.path()
        ctx.close()
        browser.close()
    shutil.move(video, out / "studio.webm")
    shutil.rmtree(out / "raw", ignore_errors=True)
    (out / "marks.json").write_text(json.dumps(r.marks, indent=1), encoding="utf-8")
    print(f"{out / 'studio.webm'}  {r.marks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
