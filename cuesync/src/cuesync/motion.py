"""Measured screen-space hand strokes, independent of voice and element style.

These are camera-compensated 2-D gestures, not estimates of ground contact or
metric 3-D punches. Observation gaps reset the detector instead of bridging them.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Stroke:
    frame: int
    kind: str
    direction: tuple[float, float]
    strength: float
    speed: float  # hand lengths per second


@dataclass
class Motion:
    velocity: np.ndarray  # px/second in each frame's camera coordinates
    valid: np.ndarray
    events: list[Stroke]

    def receipt(self):
        return [{"frame":e.frame,"kind":e.kind,"direction":list(e.direction),
                 "strength":round(e.strength,3),"hand_lengths_per_second":round(e.speed,3)}
                for e in self.events]


def analyze(track: dict, camera=None) -> Motion:
    from .bending import camera_between

    xy=np.asarray(track['xy'],float);n=len(xy)
    size=np.maximum(np.asarray(track.get('scale',[1.]*n),float),1.)
    confidence=np.asarray(track.get('conf',[1.]*n),float)
    source=np.asarray(track.get('src',['mp']*n))
    valid=np.isfinite(xy).all(1)&np.isfinite(size)&(confidence>=.72)&~np.isin(source,['lost','hold'])
    velocity=np.zeros((n,2));speed=np.zeros(n);coherence=np.zeros(n)
    # Four measured intervals suppress a one-frame jump, even with a confident
    # detector. Compare all points in the current frame before differentiating.
    for k in range(4,n):
        if not valid[k-4:k+1].all():
            continue
        p=np.array([(camera_between(camera,j,k)@np.r_[xy[j],1.])[:2] for j in range(k-4,k+1)])
        steps=np.diff(p,axis=0)
        total=np.linalg.norm(steps,axis=1).sum()
        coherence[k]=np.linalg.norm(p[-1]-p[0])/max(total,1e-6)
        lengths=np.linalg.norm(steps,axis=1)
        if (lengths>.012*size[k]).sum()<3 or lengths.max()>max(.65*size[k],3*np.median(lengths)):
            continue
        velocity[k]=(p[-1]-p[0])*6.  # all production performances are 24 fps
        speed[k]=np.linalg.norm(velocity[k])/size[k]
    events=[];armed=True;quiet=0;last=-24
    for k in range(n):
        quiet=quiet+1 if speed[k]<.8 else 0
        if quiet>=5 and k-last>=12:
            armed=True
        if not armed or speed[k]<1.65 or speed[k]>10 or coherence[k]<.88:
            continue
        direction=velocity[k]/max(np.linalg.norm(velocity[k]),1e-8)
        kind='swipe' if abs(direction[0])>1.15*abs(direction[1]) else ('lift' if direction[1]<0 else 'slam')
        events.append(Stroke(k,kind,tuple(float(v) for v in direction),float(np.clip(speed[k]/4,.4,1)),float(speed[k])))
        armed=False;last=k
    return Motion(velocity,valid,events)
