"""Private, reproducible production renders using the original phone takes.

Reuse measured tracks and mattes; never synthesize or replace the hand movement.
Default audio is the clip itself. A named --voice explicitly replaces the cue.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import numpy as np
from cuesync.bending import make_bending_shot
from cuesync.pipeline import ShotSpec

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--clips',nargs='+',default=['IMG_6668','IMG_6658'])
    p.add_argument('--effects',nargs='+',default=['water_orb','earth_rock'])
    p.add_argument('--width',type=int,default=540)
    p.add_argument('--voice',type=Path)
    p.add_argument('--hold',nargs=2,type=float,metavar=('START','END'),help='Explicit scripted hold, in seconds')
    p.add_argument('--baseline-ref')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    if args.voice is not None and args.hold is not None:
        raise SystemExit('Choose either a voice recording or a held cue')
    sources={f.relative_to(ROOT).as_posix():sha(f) for f in [*(ROOT/'cuesync/src/cuesync').glob('*.py'),*(ROOT/'cuesync/src/cuesync/assets/stone').glob('*')]}
    baseline=[]
    if args.baseline_ref:
        for name in {e.split('_')[0] for e in args.effects}:
            filename=f'cuesync/src/cuesync/{name}.py'
            source=subprocess.check_output(['git','show',f'{args.baseline_ref}:{filename}'],cwd=ROOT)
            mod=types.ModuleType('cuesync.'+name);mod.__file__=str(ROOT/filename)
            sys.modules[mod.__name__]=mod;exec(compile(source,mod.__file__,'exec'),mod.__dict__)
            if name=='earth' and not hasattr(mod,'choreography'):
                mod.choreography=lambda *a:[]
            sources[filename]=hashlib.sha256(source).hexdigest();baseline.append(filename)
    catalog=json.loads((ROOT/'runs/phone-verification-20261022/batch/results.json').read_text())
    manifest=dict(source_sha256=sources,baseline_ref=args.baseline_ref,seed=17,width=args.width,
                  audio=('Scripted held cue + Foley; no spoken cue' if args.hold else
                         'Original iPhone audio + Foley' if args.voice is None else f'Explicit cue: {args.voice.name} + Foley'),
                  voice_sha256=None if args.voice is None else sha(args.voice),cloud_spend_usd=0,cases=[])
    for clip in args.clips:
        meta=catalog[clip];source=ROOT/meta['original'];key=meta['prepare']['id']
        cached=ROOT/'runs/phone-verification-20261022/batch/jobs'/key
        inp=args.out/'inputs'/clip;inp.mkdir(parents=True)
        t=json.loads((cached/'hand_track.json').read_text());ratio=args.width/t['W'];h=round(t['H']*ratio/2)*2
        for field in ('xy','scale','speed'):
            if field in t:t[field]=(np.asarray(t[field])*ratio).tolist()
        t['landmarks']=[None if lm is None else (np.asarray(lm)*ratio).tolist() for lm in t['landmarks']]
        t['W']=args.width;t['H']=h
        (inp/'hand_track.json').write_text(json.dumps(t))
        with np.load(cached/'hand_mask.npz') as data:
            mask=np.stack([cv2.resize(f,(args.width,h),interpolation=cv2.INTER_AREA) for f in data['alpha']])
        np.savez_compressed(inp/'hand_mask.npz',alpha=mask,fps=24,version=1)
        for effect in args.effects:
            job=args.out/(clip+'-'+effect);start=time.perf_counter();print('START',job.name,flush=True)
            point=meta['tap'][:2];tap=meta['tap'][2]
            spec=ShotSpec(source,args.voice,effect,point,seed=17,cue='hold' if args.hold else 'clip' if args.voice is None else 'voice',
                          holds=[args.hold] if args.hold else None,
                          plate_size=(args.width,h),label=effect.replace('_',' '))
            result=make_bending_shot(spec,None,job,inp/'hand_track.json',inp/'hand_mask.npz',tap_frame=tap)
            for file,digest in sources.items():
                if file not in baseline and sha(ROOT/file)!=digest:
                    raise RuntimeError(f'Source changed during render: {file}')
            subprocess.run(['ffmpeg','-v','error','-i',str(result.final),'-f','null','-'],check=True)
            manifest['cases'].append(dict(clip=clip,effect=effect,source=str(source),source_sha256=sha(source),
                track_sha256=sha(inp/'hand_track.json'),mask_sha256=sha(inp/'hand_mask.npz'),point=point,tap_frame=tap,
                seconds=round(time.perf_counter()-start,2),final=str(result.final.resolve()),sha256=sha(result.final),decode='passed',
                movement=result.receipt.get('bending',{}).get('movement')))
            (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('DONE',job.name,manifest['cases'][-1]['seconds'],flush=True)


if __name__=='__main__':
    main()
