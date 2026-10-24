"""Measured screen-space hand strokes, independent of voice and element style.

These are camera-compensated 2-D gestures, not estimates of ground contact or
metric 3-D punches. Observation gaps reset the detector instead of bridging them.
"""
from dataclasses import dataclass
import numpy as np


def bridge_short_gaps(track, plate, mask, max_gap=12):
    """Carry a visible hand through short detector gaps using measured image flow.

    SAM's mask limits features to the hand. Forward/backward agreement rejects
    disappearing features. Bridged samples cannot trigger gestures: confidence
    remains below the stroke threshold. Never extrapolate without visual support.
    """
    import cv2
    if mask is None:
        return track
    result=dict(track);n=min(len(plate),len(track['xy']))
    result['xy']=[list(v) for v in track['xy']]
    result['src']=list(track.get('src',['mp']*n))
    result['conf']=list(track.get('conf',[1.]*n))
    result['landmarks']=list(track.get('landmarks',[None]*n))
    result['scale']=list(track['scale'])
    result['dir']=[list(v) for v in track.get('dir',[[0.,-1.]]*n)]
    bridged=[];gap=0
    for k in range(1,n):
        if track['src'][k] not in ('lost','hold'):
            gap=0;continue
        gap+=1
        if gap>max_gap or result['src'][k-1] in ('lost','hold'):
            continue
        previous=cv2.cvtColor(plate[k-1],cv2.COLOR_RGB2GRAY)
        current=cv2.cvtColor(plate[k],cv2.COLOR_RGB2GRAY)
        area=cv2.erode(mask[k-1],np.ones((5,5),np.uint8))
        yy,xx=np.mgrid[:area.shape[0],:area.shape[1]];x,y=result['xy'][k-1]
        size=max(float(result['scale'][k-1]),1.)
        area=np.where((xx-x)**2+(yy-y)**2<(1.5*size)**2,area,0).astype(np.uint8)
        transform=None
        # A fast lift can smear skin detail across a small LK window. Retry at
        # a coarser image scale, requiring a spatially distributed consensus.
        for window,blur in ((25,0),(61,5)):
            prev=cv2.GaussianBlur(previous,(blur,blur),0) if blur else previous
            curr=cv2.GaussianBlur(current,(blur,blur),0) if blur else current
            points=cv2.goodFeaturesToTrack(prev,80,.015,7,mask=area)
            if points is None or len(points)<4:
                continue
            nxt,ok,_=cv2.calcOpticalFlowPyrLK(prev,curr,points,None,winSize=(window,window),maxLevel=4)
            back,valid,_=cv2.calcOpticalFlowPyrLK(curr,prev,nxt,None,winSize=(window,window),maxLevel=4)
            coords=np.rint(nxt[:,0]).astype(int)
            inside=(coords[:,0]>=0)&(coords[:,0]<area.shape[1])&(coords[:,1]>=0)&(coords[:,1]<area.shape[0])
            coords[:,0]=np.clip(coords[:,0],0,area.shape[1]-1);coords[:,1]=np.clip(coords[:,1],0,area.shape[0]-1)
            keep=(ok.ravel()>0)&(valid.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
            keep&=inside&(mask[k,coords[:,1],coords[:,0]]>128)
            if keep.sum()<4:
                continue
            steps=(nxt-points).reshape(-1,2)[keep];median=np.median(steps,axis=0)
            agree=np.linalg.norm(steps-median,axis=1)<max(3.,.06*size)
            support=points[:,0][keep][agree]
            if (len(support)>=max(4,int(np.ceil(.75*keep.sum())))
                and np.linalg.norm(np.ptp(support,axis=0))>=.08*size
                and np.linalg.norm(median)<=size):
                transform=np.column_stack([np.eye(2),np.median(steps[agree],axis=0)])
                zoom=1.;break
            candidate,inliers=cv2.estimateAffinePartial2D(points[:,0][keep],nxt[:,0][keep],
                method=cv2.RANSAC,ransacReprojThreshold=max(2.,.027*size))
            if candidate is None or inliers is None:
                continue
            support=points[:,0][keep][inliers.ravel()>0]
            zoom=np.sqrt(abs(np.linalg.det(candidate[:,:2])))
            angle=abs(np.arctan2(candidate[1,0],candidate[0,0]))
            delta=candidate@np.r_[x,y,1.]-[x,y]
            if (len(support)>=max(4,int(np.ceil(.55*keep.sum())))
                and np.linalg.norm(np.ptp(support,axis=0))>=.08*size
                and .65<=zoom<=1.4 and angle<.65 and np.linalg.norm(delta)<=size):
                transform=candidate;break
        if transform is None:
            continue
        result['xy'][k]=(transform@np.r_[x,y,1.]).tolist()
        result['scale'][k]=size*zoom
        result['dir'][k]=(transform[:,:2]@result['dir'][k-1]/zoom).tolist()
        lm=result['landmarks'][k-1]
        if lm is not None:
            lm=np.asarray(lm).copy();lm[:,:2]=lm[:,:2]@transform[:,:2].T+transform[:,2]
            lm[:,2]*=zoom;result['landmarks'][k]=lm.tolist()
        result['src'][k]='flow';result['conf'][k]=.60;bridged.append(k)
    result['flow_bridge_frames']=bridged
    return result


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
