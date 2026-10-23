"""Build private bright/dim element reels and matched before/after comparisons.

The reel shows the first 3.5 seconds of each source at its original 24 fps.
No time warp, generated replacement footage, or speed-up. Audio is the left
bright-room panel's actual final mix. Full original clips remain in the batch.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from install_element_gallery import NAMES


@lru_cache(maxsize=8)
def font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def panel(image, size):
    # Same fixed crop for every version: retain the wall, fingertips and palm.
    h,w=image.shape[:2]
    top=round(h*.17)
    bottom=min(h,round(h*.94))
    crop=Image.fromarray(cv2.cvtColor(image[top:bottom],cv2.COLOR_BGR2RGB))
    crop.thumbnail(size,Image.Resampling.LANCZOS)
    return crop


def clip(paths, labels, title, audio, target, columns):
    width,height=(1440,1080) if columns==2 else (1920,1080)
    captures=[cv2.VideoCapture(str(p)) for p in paths]
    n=min(84,*(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures))
    if n<24: raise ValueError(f"Invalid/too short clip in {paths}")
    cmd=["ffmpeg","-v","error","-y","-f","rawvideo","-pixel_format","rgb24","-video_size",f"{width}x{height}",
         "-framerate","24","-i","pipe:0","-i",str(audio),"-map","0:v:0","-map","1:a:0",
         "-af",f"apad,atrim=end={n/24}","-c:v","libx264","-preset","fast","-crf","20","-threads","2",
         "-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-movflags","+faststart",str(target)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for frame in range(n):
            canvas=Image.new("RGB",(width,height),(13,15,18));draw=ImageDraw.Draw(canvas)
            draw.text((34,20),title,font=font(36),fill=(241,237,228))
            cell=width//columns
            for i,(capture,label) in enumerate(zip(captures,labels)):
                ok,bgr=capture.read()
                if not ok: raise RuntimeError(f"Decode ended early: {paths[i]}")
                draw.text((i*cell+28,75),label,font=font(23),fill=(180,188,197))
                tile=panel(bgr,(cell-28,882))
                canvas.paste(tile,(i*cell+(cell-tile.width)//2,116))
            draw.text((34,1025),"Original speed / 24 fps. Audio: bright-room performance. Sample cues; not a live microphone capture.",
                      font=font(19),fill=(153,161,171))
            process.stdin.write(np.asarray(canvas).tobytes())
    finally:
        process.stdin.close()
        for c in captures:c.release()
    if process.wait(): raise RuntimeError(f"Video encoding failed: {target}")
    subprocess.run(["ffmpeg","-v","error","-i",str(target),"-f","null","-"],check=True)


def join(parts,target):
    listing=target.with_suffix(".concat.txt")
    listing.write_text("\n".join("file '"+str(p.resolve()).replace("\\","/").replace("'","'\\''")+"'" for p in parts),encoding="utf-8")
    subprocess.run(["ffmpeg","-v","error","-y","-f","concat","-safe","0","-i",str(listing),"-c","copy",
                    "-movflags","+faststart",str(target)],check=True)
    subprocess.run(["ffmpeg","-v","error","-i",str(target),"-f","null","-"],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after",type=Path,required=True)
    p.add_argument("--before",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.after/"manifest.json").read_text())
    if len(manifest["cases"])!=16: raise ValueError("Complete sixteen-shot suite required")
    before=json.loads((args.before/"manifest.json").read_text())
    assert manifest["width"]==before["width"] and manifest["seed"]==before["seed"], "Comparison resolution/seed differ"
    current={(c["case"],c["effect"]):c for c in manifest["cases"]}
    for old in before["cases"]:
        new=current[(old["case"],old["effect"])]
        for field in ("plate_sha256","voice_sha256","track_source_sha256","mask_source_sha256","offset_s"):
            assert old[field]==new[field],f"Comparison input differs: {field}"
    parts=[];comparisons=[]
    for effect,title in NAMES.items():
        paths=[args.after/f"{case}-{effect}"/"composite.mp4" for case in ("bright","dark")]
        audio=args.after/f"bright-{effect}"/"final.mp4"
        result=args.out/f"{effect}.mp4"
        clip(paths,["Bright room","Dim room"],title,audio,result,2)
        parts.append(result)
        if effect in ("air_ball","water_orb","lightning_ball"):
            paths=[root/f"{case}-{effect}"/"composite.mp4" for case in ("bright","dark") for root in (args.before,args.after)]
            comparison=args.out/f"{effect}-before-after.mp4"
            clip(paths,["Bright / before","Bright / after","Dim / before","Dim / after"],title+" / before and after",audio,comparison,4)
            comparisons.append(comparison)
    join(parts,args.out/"four-elements-eight-forms.mp4")
    join(comparisons,args.out/"before-after.mp4")
    poster=Image.new("RGB",(2048,760),(13,15,18));draw=ImageDraw.Draw(poster)
    for i,(effect,case,frame) in enumerate((("air_ball","bright",54),("water_orb","bright",37),
                                          ("lightning_ball","dark",53),("earth_rock","bright",50))):
        capture=cv2.VideoCapture(str(args.after/f"{case}-{effect}"/"composite.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=capture.read();capture.release()
        if not ok:raise RuntimeError("Poster frame missing")
        h,w=bgr.shape[:2]
        crop=ImageOps.fit(Image.fromarray(cv2.cvtColor(bgr[round(.23*h):round(.94*h)],cv2.COLOR_BGR2RGB)),(500,680),method=Image.Resampling.LANCZOS)
        poster.paste(crop,(i*512+6,60));draw.text((i*512+18,14),NAMES[effect],font=font(28),fill="white")
    poster.save(args.out/"elements-polished.jpg",quality=94)
    outputs={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.iterdir() if f.suffix in (".mp4",".jpg")}
    (args.out/"review-manifest.json").write_text(json.dumps({"inputs":manifest,"outputs":outputs,
        "edit":"First 84 frames at 24 fps; identical fixed framing; audio from the bright after panel; original full clips retained."},indent=2))
    print(f"Created {len(parts)} paired forms, three matched comparisons, poster and reels in {args.out}")


if __name__=="__main__":main()
