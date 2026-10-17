"""Does a take land where it was performed? The studio's microphone is replaced by a synthetic one that makes a
tone burst at the moment the clip shows a chosen time (the page's own frame callback), and the take is analysed
by the server like any other; the noise must be found at that time on the clip.

    python tools/check_take_timing.py --plate "Candle jar" --at 1.0 1.5 2.2 --chromium <chrome.exe>

Needs the studio (npm run dev) and the render API (cuesync serve --instant-only is enough) running.
"""

from __future__ import annotations

import argparse
import json
import sys

from playwright.sync_api import sync_playwright

FAKE_MIC = """
(() => {
  const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    if (!c || !c.audio || c.video) return real(c);
    const ctx = new AudioContext();
    await ctx.resume();
    const dest = ctx.createMediaStreamDestination();
    const watch = () => {
      const v = document.querySelector('video.plate');
      if (!v || v.paused || !v.requestVideoFrameCallback) return setTimeout(watch, 20);
      const cb = (now, md) => {
        if (md.mediaTime >= window.__clickAt) {
          const o = ctx.createOscillator(), g = ctx.createGain();
          o.frequency.value = 160;
          g.gain.setValueAtTime(0.7, ctx.currentTime);
          g.gain.setTargetAtTime(0, ctx.currentTime + 0.3, 0.04);
          o.connect(g).connect(dest);
          o.start();
          o.stop(ctx.currentTime + 0.6);
          window.__clickMedia = md.mediaTime;
          return;
        }
        v.requestVideoFrameCallback(cb);
      };
      v.requestVideoFrameCallback(cb);
    };
    watch();
    return dest.stream;
  };
})();
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:3000")
    ap.add_argument("--plate", default="Candle jar")
    ap.add_argument("--at", type=float, nargs="+", default=[1.0, 1.5, 2.2], help="clip times to make the noise at")
    ap.add_argument("--wait", type=float, default=2.0, help="seconds between switching the microphone on and performing")
    ap.add_argument("--chromium")
    a = ap.parse_args()
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chromium, args=["--autoplay-policy=no-user-gesture-required"])
        for at in a.at:
            ctx = browser.new_context(viewport={"width": 1280, "height": 800}, permissions=["microphone"])
            ctx.add_init_script(f"window.__clickAt = {at};" + FAKE_MIC)
            page = ctx.new_page()
            page.goto(a.url)
            page.get_by_text(a.plate, exact=True).first.click()
            page.get_by_role("button", name="Use my microphone").click()
            page.wait_for_timeout(int(a.wait * 1000))
            page.get_by_role("button", name="Perform").click()
            page.get_by_role("heading", name="4. Make the shot").wait_for(timeout=60000)
            clicked = page.evaluate("window.__clickMedia")
            with page.expect_response(lambda r: r.url.endswith("/api/jobs") and r.request.method == "POST") as resp:
                page.get_by_role("radio", name="Instant").click()
                page.get_by_role("button", name="Make the shot").click()
            job = resp.value.json()
            page.locator("video.result").wait_for(timeout=180000)
            receipt = page.evaluate(f"fetch('/api/jobs/{job['id']}').then(r => r.json())")["receipt"]
            found = receipt["voice"]["onsets_s"][0]
            err = (found - clicked) * 1000
            errors.append(err)
            print(f"noise made at clip {clicked:.3f} s, found at {found:.3f} s: {err:+.0f} ms", flush=True)
            ctx.close()
        browser.close()
    print(json.dumps({"errors_ms": [round(e) for e in errors], "worst_ms": round(max(map(abs, errors)))}))
    return 0 if max(map(abs, errors)) <= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
