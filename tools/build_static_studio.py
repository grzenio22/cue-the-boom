"""Build the studio as static files that `cuesync serve` itself serves (no Node.js needed to run it): a Next.js static
export of web/ copied into cuesync/src/cuesync/studio_static/, which ships inside the cuesync package.

    python tools/build_static_studio.py        # needs Node 24 (only to build; running it needs none)

The static studio has no judging password (that is web/proxy.ts, a server feature): a hosted, password-protected
studio runs the Next.js server instead (infra/pod/serve-app.sh, deploy/space).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "cuesync" / "src" / "cuesync" / "studio_static"


def main() -> int:
    npx = shutil.which("npx")
    if npx is None:
        print("needs Node 24 (npx on the PATH) to build", file=sys.stderr)
        return 1
    if not (WEB / "node_modules").exists():
        subprocess.run([shutil.which("npm"), "install"], cwd=WEB, check=True)
    proxy, off = WEB / "proxy.ts", WEB / "proxy.ts.static-off"
    proxy.rename(off)  # a static export can't have a request proxy (the judging password): set it aside meanwhile
    try:
        subprocess.run([npx, "next", "build"], cwd=WEB, env=os.environ | {"CUESYNC_STATIC": "1"}, check=True)
    finally:
        off.rename(proxy)
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(WEB / "out", OUT)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"{OUT}: {size:.1f} MB; cuesync serve now serves the studio at its own address")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
