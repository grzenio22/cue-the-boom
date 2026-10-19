"""Record the studio being used, for the demo: pick a shot, cue it, make the shot, re-time it.

    python tools/demo/record_studio.py --plate "Toy dinosaur, night" --say samples/voices/fwoooom.wav -o runs/demo/studio
    python tools/demo/record_studio.py --plate "Candle jar" --sample "pf." -o runs/demo/studio_offline
    python tools/demo/record_studio.py --upload my_clip.mp4 --point 0.43,0.30 --object "a jack-o'-lantern" \
        --say my_take.wav -o runs/demo/studio_upload

--say: the browser's microphone plays that file (Chromium's fake capture device), so the take goes through the
same microphone path a person's voice does; --sample picks a sample noise; --clip uses the clip's own sound.
--upload: your own clip instead of a sample, tapped at --point (x,y as fractions of the picture) and, for a scene
shot, said to be --object (one of the studio's "What is it?" choices).
Needs Playwright (pip install playwright) and a Chromium (--chromium, else Playwright's own). Writes studio.webm
with a visible cursor and marks.json (seconds into the video of each step), so the edit can cut the render wait.
--hd: the page at 960x720 and twice the pixels, captured frame by frame as it paints (studio.mp4, 1920x1440, text
as sharp as on a laptop screen, for the edit to crop) instead of Playwright's video; needs ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import re
import json
import shutil
import subprocess
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


class Screencast:
    """The page's own frames at its device pixels as it paints them, each kept with the second it came."""

    def __init__(self, ctx, page: Page, t0: float, out: Path):
        self.out, self.t0, self.times = out, t0, []
        out.mkdir(parents=True, exist_ok=True)
        self.cdp = ctx.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self.frame)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "maxWidth": 1920, "maxHeight": 1440})

    def frame(self, ev: dict) -> None:
        (self.out / f"{len(self.times):05d}.jpg").write_bytes(base64.b64decode(ev["data"]))
        self.times.append(time.perf_counter() - self.t0)
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    def save(self, video: Path) -> None:
        """24 fps from the first paint, each frame held until the next one came."""
        self.cdp.send("Page.stopScreencast")
        ts = self.times + [self.times[-1] + 1.0]
        name = lambda k: (self.out / f"{k:05d}.jpg").resolve().as_posix()  # noqa: E731
        lines = [f"file '{name(k)}'\nduration {ts[k + 1] - ts[k]:.4f}" for k in range(len(self.times))]
        listing = self.out / "frames.txt"
        listing.write_text("\n".join(lines) + f"\nfile '{name(len(self.times) - 1)}'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-vf",
                        "scale=1920:1440:force_original_aspect_ratio=decrease,pad=1920:1440:-1:-1,fps=24",
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(video)], check=True)
        # the frames are timed from the recorder's clock, as the marks are: the video starts at the first paint
        self.first = self.times[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:3000")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--plate", help="the sample shot's title, as the studio shows it")
    src.add_argument("--upload", help="your own clip, uploaded through 'Use your own clip'")
    ap.add_argument("--point", help="with --upload: where it comes from, x,y as fractions of the picture")
    ap.add_argument("--object", help="with --upload: what it is, one of the studio's 'What is it?' choices")
    cue = ap.add_mutually_exclusive_group(required=True)
    cue.add_argument("--say", help="a WAV the fake microphone plays while you perform")
    cue.add_argument("--sample", help="a sample noise's title")
    cue.add_argument("--clip", action="store_true", help="lock it to the clip's own sound")
    ap.add_argument("--draw", help='draw instead of tapping: one stroke as JSON points on the plate, "[[x, y], ...]"')
    ap.add_argument("--brush", default="medium", help="the brush to draw with, as the studio labels it")
    ap.add_argument("--describe", help="type this into 'Say what happens'")
    ap.add_argument("--retime", type=float, default=0.03, help="drag the hit by this share of the ribbon (0: don't)")
    ap.add_argument("--hand", action="store_true", help="the plate is a hand: its spot (GPU offline: Perform it) or "
                                                       "tap --point and drag to aim at --aim degrees (live)")
    ap.add_argument("--frame", type=int, help="with --upload --hand: the frame to tap your hand on (24 fps)")
    ap.add_argument("--aim", type=float, default=0.0, help="with --hand, live: the aim, degrees clockwise from up")
    ap.add_argument("--again", help="after the shot: Perform it again with this sample noise (no new render)")
    ap.add_argument("--chromium", help="a Chromium executable (default: Playwright's)")
    ap.add_argument("--hd", action="store_true", help="1920x1080 frames as the page paints them (see above)")
    ap.add_argument("-o", "--out", default="runs/demo/studio")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = ["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
    if a.say:
        args += ["--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={Path(a.say).resolve()}%noloop"]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chromium, args=args)
        if a.hd:
            ctx = browser.new_context(viewport={"width": 960, "height": 720}, device_scale_factor=2, permissions=["microphone"])
        else:
            ctx = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1,
                                      record_video_dir=str(out / "raw"), record_video_size={"width": 1280, "height": 800},
                                      permissions=["microphone"])
        ctx.add_init_script(CURSOR)
        page = ctx.new_page()
        r = Recorder(page)
        cast = Screencast(ctx, page, r.t0, out / "frames") if a.hd else None
        page.goto(a.url)
        page.get_by_role("heading", name="1. Pick a shot").wait_for(timeout=30000)
        r.pause(1.5)
        r.mark("pick")
        if a.upload:  # your own clip: upload it, then tap where it comes from
            label = page.get_by_text("Use your own clip", exact=True)
            r.click(label, pause=0.4)
            page.locator('input[type="file"]').set_input_files(str(Path(a.upload).resolve()))
            page.get_by_role("heading", name="2. Where does it come from?").wait_for(timeout=30000)
            r.pause(1.2)
            if a.hand:  # your hand is the source, not a thing in the shot
                r.click(page.get_by_role("radio", name="Your hand"), pause=1.0)
            elif not a.draw:
                tap = page.locator("section").filter(has=page.get_by_role("heading", name="2. Where does it come from?")).locator("video").first
                tap.scroll_into_view_if_needed()
                page.wait_for_function("v => v.videoWidth > 0", arg=tap.element_handle(), timeout=30000)
                box = tap.bounding_box()
                vw, vh = tap.evaluate("v => [v.videoWidth, v.videoHeight]")
                k = min(box["width"] / vw, box["height"] / vh)  # object-fit: contain
                cw, ch = vw * k, vh * k
                x, y = (float(v) for v in a.point.split(","))
                px, py = box["x"] + (box["width"] - cw) / 2 + x * cw, box["y"] + (box["height"] - ch) / 2 + y * ch
                page.mouse.move(px, py, steps=18)
                r.pause(0.3)
                r.mark("tap")
                page.mouse.click(px, py)
                r.pause(1.0)
        else:
            page.get_by_text(a.plate, exact=True).first.wait_for(timeout=30000)
            r.click(page.get_by_text(a.plate, exact=True).first, pause=1.2)
            if a.hand and page.get_by_role("button", name="Use my microphone").count():
                r.mark("hand")  # the sample hand comes with its spot: straight to the cue
            else:
                r.click(page.get_by_role("button", name="Move it"), pause=1.5)  # the spot comes with the sample: show it
        if a.hand and "hand" not in r.marks:  # your hand: the sample's spot made ahead of time, or tap it and drag to aim
            perform = page.get_by_role("button", name="Perform it", exact=True)
            if perform.count():
                r.mark("hand")
                r.click(perform, pause=1.0)
            else:
                import math

                if a.frame is not None:  # a frame where the hand is clear
                    page.locator('input[type="range"]').first.fill(str(a.frame))
                    r.pause(0.8)
                aim = page.locator("section").filter(has=page.get_by_role("heading", name="2. Where does it come from?")).locator("video").first
                page.wait_for_function("v => v.videoWidth > 0", arg=aim.element_handle(), timeout=30000)
                y_ = float(a.point.split(",")[1])  # the spot in the middle of the screen (a phone clip is taller than it)
                aim.evaluate("(el, y) => { const b = el.getBoundingClientRect(); scrollBy({top: b.top + y * b.height - innerHeight / 2, behavior: 'instant'}); }", y_)
                r.pause(0.6)
                box = aim.bounding_box()
                x, y = (float(v) for v in a.point.split(","))
                px, py = box["x"] + x * box["width"], box["y"] + y * box["height"]
                page.mouse.move(px, py, steps=18)
                r.pause(0.3)
                r.mark("hand")
                page.mouse.down()
                page.mouse.move(px + 90 * math.sin(math.radians(a.aim)), py - 90 * math.cos(math.radians(a.aim)), steps=16)
                page.mouse.up()
                r.pause(1.2)
                r.click(page.get_by_role("button", name="That's my hand"), pause=0.5)
                # the studio follows your hand through the clip and draws its path: watch it, then go on
                ok = page.get_by_role("button", name=re.compile("^(Looks right|Go on with it)$"))
                mic = page.get_by_role("button", name="Use my microphone")
                r.mark("following")
                for _ in range(600):
                    if ok.count() or mic.count():
                        break
                    r.pause(0.5)
                if ok.count():
                    r.mark("followed")  # the hand's path in view (the buttons under it may be off the screen)
                    aim.evaluate("(el, y) => { const b = el.getBoundingClientRect(); scrollBy({top: b.top + y * b.height - innerHeight / 2, behavior: 'instant'}); }", y_)
                    r.pause(1.5)
                    play = page.get_by_role("button", name="Play it")
                    if play.count():
                        play.evaluate("b => b.click()")  # (clicked where it is: a pointer click scrolls the hand away)
                        r.mark("play")
                        r.pause(4.0)
                    r.click(ok, pause=1.0)
        elif a.draw:
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
        elif not a.hand:
            r.click(page.get_by_role("button", name="That's the spot"), pause=1.0)
        r.mark("cue")
        if a.say:
            r.click(page.get_by_role("button", name="Use my microphone"), pause=1.5)
            # the playing clip and the level meter in view before the take starts, not after
            page.locator("video.plate").evaluate("el => el.scrollIntoView({block: 'center', behavior: 'instant'})")
            r.pause(0.6)
            r.click(page.get_by_role("button", name="Perform"), pause=0.0)
            r.mark("performing")
            page.get_by_role("heading", name="4. Make the shot").wait_for(timeout=60000)
        elif a.clip:
            r.click(page.get_by_role("radio", name="The clip's own sound"), pause=1.0)
            r.click(page.get_by_role("button", name="Lock it to my clip's sound"), pause=1.0)
        else:
            r.click(page.locator("button", has_text=a.sample).first, pause=1.0)
        if a.object:  # your own clip: what it is
            r.click(page.get_by_role("radio", name=a.object, exact=True), pause=0.8)
        if a.describe:
            field = page.get_by_label("Say what happens")
            r.click(field, pause=0.3)
            page.keyboard.type(a.describe, delay=35)
        r.pause(1.5)
        r.mark("make")
        r.click(page.get_by_role("button", name="Make the shot"), pause=0.5)
        page.locator("video.result, p.chip.bad").first.wait_for(timeout=900000)
        if page.locator("p.chip.bad").count():  # the shot failed: say why, don't wait for a result that won't come
            raise SystemExit(f"the shot failed: {page.locator('p.chip.bad').first.inner_text()}")
        r.mark("done")
        page.locator("video.result").evaluate("el => el.scrollIntoView({block: 'center', behavior: 'instant'})")  # the whole shot in view
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
        if a.again:  # a new rhythm on the same render
            r.click(page.get_by_role("button", name="Perform it again"), pause=1.2)
            r.mark("again")
            r.click(page.locator("button", has_text=a.again).first, pause=0.5)
            page.locator("progress").first.wait_for(timeout=60000)  # the replay starts...
            page.wait_for_function("() => !document.querySelectorAll('progress').length", timeout=900000)  # ...and ends
            r.mark("again_done")
            page.locator("video.result").last.scroll_into_view_if_needed()
            r.pause(6.0)
        r.mark("end")
        if cast:
            cast.save(out / "studio.mp4")
            r.marks = {k: round(v - cast.first, 2) for k, v in r.marks.items()}  # seconds into the video
        else:
            video = page.video.path()
        ctx.close()
        browser.close()
    if cast:
        shutil.rmtree(out / "frames", ignore_errors=True)
    else:
        shutil.move(video, out / "studio.webm")
        shutil.rmtree(out / "raw", ignore_errors=True)
    (out / "marks.json").write_text(json.dumps(r.marks, indent=1), encoding="utf-8")
    print(f"{out / ('studio.mp4' if cast else 'studio.webm')}  {r.marks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
