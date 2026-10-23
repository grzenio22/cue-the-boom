"""Earth bending: tracked, rotating stone geometry with reflected light and falling grit.

The mesh is three dimensional; depth tests order its faces, orbiting fragments and
the photographed hand. Material detail lives in object coordinates, so it rotates
with the stone. This is art-directed CPU geometry, not a simulated fracture solver.
"""
from __future__ import annotations

import math
from itertools import combinations
from dataclasses import dataclass

import cv2
import numpy as np

from .elements import Timeline
from .transfer import lin, gam
from .water import hand_of

FORMS = ("rock", "shatter")


@dataclass
class Stone:
    vertices: np.ndarray
    faces: np.ndarray
    phase: np.ndarray
    colour: np.ndarray


def stone(seed: int) -> Stone:
    rng = np.random.default_rng(seed)
    phi=(1+math.sqrt(5))/2
    pts=np.array([(0,a,b*phi) for a in (-1,1) for b in (-1,1)]+
                 [(a,b*phi,0) for a in (-1,1) for b in (-1,1)]+
                 [(b*phi,0,a) for a in (-1,1) for b in (-1,1)],float)
    faces=[]
    for ids in combinations(range(12),3):
        a,b,c=pts[list(ids)]
        distances=(pts-a)@np.cross(b-a,c-a)
        if np.all(distances>=-1e-8) or np.all(distances<=1e-8):
            faces.append(ids)
    vertices=list(pts)
    midpoints={}
    def midpoint(a,b):
        key=tuple(sorted((a,b)))
        if key not in midpoints:
            midpoints[key]=len(vertices)
            vertices.append((pts[a]+pts[b])/2)
        return midpoints[key]
    refined=[]
    for a,b,c in faces:
        ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
        refined.extend([(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)])
    pts=np.array(vertices)
    pts/=np.linalg.norm(pts,axis=1,keepdims=True)
    pts*=rng.uniform(.77,1.13,(len(pts),1))*np.array([1.08,.87,.94])
    faces=np.array(refined)
    return Stone(pts, faces, rng.uniform(0, 6.28, 5), np.array([.54, .49, .41]) * rng.uniform(.9, 1.1))


def rotation(t: float) -> np.ndarray:
    a, b = t*.57, t*.33+.4
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]) @ np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])


def raster(mesh: Stone, center, radius, turn, rgb, alpha, depth, light, irradiance):
    """Opaque, textured mesh into a shared local depth buffer, nearest Z wins."""
    H, W = alpha.shape
    verts = mesh.vertices @ turn.T * radius + center
    for face in mesh.faces:
        abc = verts[face]
        normal = np.cross(abc[1]-abc[0], abc[2]-abc[0])
        normal /= max(np.linalg.norm(normal), 1e-9)
        if normal @ (abc.mean(0)-center) < 0:
            normal = -normal
        if normal[2] <= 0:
            continue
        x0, y0 = np.maximum(np.floor(abc[:,:2].min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(abc[:,:2].max(0)).astype(int)+1, [W,H])
        if x1<=x0 or y1<=y0:
            continue
        yy, xx = np.mgrid[y0:y1,x0:x1].astype(np.float32)
        a,b,c=abc
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-8:
            continue
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        z=u*a[2]+v*b[2]+(1-u-v)*c[2]
        covered=(u>=0)&(v>=0)&(u+v<=1)&(z>depth[y0:y1,x0:x1])
        if not covered.any():
            continue
        obj=u[...,None]*mesh.vertices[face[0]]+v[...,None]*mesh.vertices[face[1]]+(1-u-v)[...,None]*mesh.vertices[face[2]]
        X,Y,Z=np.moveaxis(obj,-1,0)
        ph=mesh.phase
        strata=np.sin(17*Y+3*np.sin(3*X+ph[0])+2*Z)
        grain=(np.sin(53*X+31*Y+ph[1])*np.sin(47*Z-19*X+ph[2])+
               .5*np.sin(137*X+81*Y-93*Z+ph[3]))/1.5
        vein=np.exp(-((np.sin(7*X+5*Z+2*np.sin(4*Y)+ph[4]))/.10)**2)
        material=np.clip(.83+.12*strata+.12*grain-.22*vein,.35,1.1)
        # Broad rounded normals and a smaller facet contribution preserve rugged
        # edges without making every triangle read as a low-poly game asset.
        smooth = obj @ turn.T
        smooth /= np.maximum(np.linalg.norm(smooth,axis=-1,keepdims=True),1e-8)
        facing = np.maximum(0, .65*(smooth@light)+.35*float(normal@light))
        shade=.32+.68*facing
        colour=material[...,None]*mesh.colour*irradiance*shade[...,None]
        rgb[y0:y1,x0:x1][covered]=colour[covered]
        alpha[y0:y1,x0:x1][covered]=1
        depth[y0:y1,x0:x1][covered]=z[covered]


def _events(tl, form):
    if form == "shatter":
        return [(int(f), min(tl.n,int(f)+15), float(z)) for f,z in zip(tl.hits,tl.sizes) if 0<=f<tl.n]
    active=np.asarray(tl.energy)>.25
    for a,b in tl.holds:
        active[max(0,a):min(tl.n,b)]=True
    edges=np.diff(np.r_[False,active,False].astype(int))
    return [(int(a),int(b),float(np.max(tl.energy[a:b],initial=.5)))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def _flight_center(hand, anchor, k, camera, *, falling=True):
    """Released mass inherits hand velocity once, then moves with the camera."""
    from .bending import camera_between

    s=float(hand.s[anchor]);p=hand.p[anchor]
    previous=max(0,anchor-1)
    previous_in_frame=camera_between(camera,previous,anchor) @ np.r_[hand.p[previous],1.]
    velocity=(p-previous_in_frame[:2])*24
    velocity*=min(1.,3*s/max(float(np.linalg.norm(velocity)),1e-8))
    age=(k-anchor)/24
    xy=p+np.array([.1*s*age,-.65*s+(1.8*s*age*age if falling else 0.)])+velocity*age
    transform=camera_between(camera,anchor,k)
    xy=(transform @ np.r_[xy,1.])[:2]
    zoom=math.sqrt(abs(float(np.linalg.det(transform[:2,:2]))))
    return np.r_[xy,float(hand.depth(anchor,p[0],p[1],fingers=False))+.22*s],s*zoom


def render(plate, track, mask, timeline: Timeline, form: str, seed=0, progress=None):
    if form not in FORMS:
        raise ValueError(f"earth form must be one of {FORMS}")
    n,h,w=plate.shape[:3]
    if timeline.n!=n or (mask is not None and mask.shape!=(n,h,w)):
        raise ValueError("plate, timeline and mask dimensions must agree")
    events=_events(timeline,form)
    out=plate.copy()
    if not events:
        return out
    hand=hand_of(track,n,plate)
    meshes=[stone(seed+j*101) for j in range(13)]
    rng=np.random.default_rng(seed+417)
    orbit=rng.uniform(0,2*np.pi,12)
    dust=rng.normal(size=(100,3))
    dust[:,1]=np.abs(dust[:,1])*.6
    dust_size=rng.uniform(.003,.016,100)
    light=np.array([-.45,-.72,.62]);light/=np.linalg.norm(light)
    # Sample existing incident room light. Stone cannot light an unlit scene.
    env=lin(plate[::max(1,n//6),:max(1,h//2):8,::8])
    irradiance=np.maximum(np.percentile(env,88,axis=(0,1,2)),.001)
    irradiance=np.clip(irradiance*.85+.15*irradiance.mean(),.001,1)
    say=progress or (lambda *args:None)
    for k in range(n):
        presence=float(hand.presence[k])
        if presence<=0:
            continue
        live=[(a,b,g) for a,b,g in events if a<=k<min(n,b+40)]
        if not live:
            continue
        P=lin(plate[k])
        for event,(a,b,gain) in enumerate(live):
            s=float(hand.s[k]); age=(k-a)/24; release=max(0,(k-b)/24)
            fade=max(0,1-release/1.6)*presence
            if fade<=0:
                continue
            grow=min(1,(k-a+1)/7)
            grow=grow*grow*(3-2*grow)
            r=s*(.40+.10*gain)*grow
            if r<.2:
                continue
            p=hand.p[k]
            held=np.array([p[0]+.08*s*math.sin(age*1.5),p[1]-.65*s-.025*s*math.sin(age*3),
                           float(hand.depth(k,p[0],p[1],fingers=False))+.22*s])
            burst=form=="shatter" and k>=a+6
            spread=max(0,(k-a-6)/24) if burst else 0
            if release or burst:
                anchor=min(a+5 if burst else b-1,n-1)
                held,s=_flight_center(hand,anchor,k,timeline.camera,falling=not burst)
                r=s*(.40+.10*gain)
            # A local working box bounds geometry and dust memory at export resolution.
            reach=(2.1+min(3,3*spread)+min(2,2*release))*s
            x0,y0=np.maximum(np.floor(held[:2]-reach).astype(int),0)
            x1,y1=np.minimum(np.ceil(held[:2]+reach).astype(int)+1,[w,h])
            if x1<=x0 or y1<=y0:
                continue
            bh,bw=y1-y0,x1-x0
            C=np.zeros((bh,bw,3),np.float32);A=np.zeros((bh,bw),np.float32);Z=np.full((bh,bw),-np.inf,np.float32)
            center=held-np.array([x0,y0,0])
            if not burst:
                raster(meshes[0],center,r,rotation(age),C,A,Z,light,irradiance)
            for j in range(1,13):
                phase=orbit[j-1]+age*(.45+.025*j)
                rad=s*(.65+.10*(j%3))
                off=np.array([math.cos(phase)*rad,math.sin(phase)*rad*.45,math.sin(phase)*rad*.7])
                if burst:
                    off*=.45+spread*3
                    off[1]+=s*2.0*spread*spread
                rr=r*((.37+.045*(j%3)) if burst else (.13+.035*(j%3)))
                raster(meshes[j],center+off,rr,rotation(age*(1+j*.2)+j),C,A,Z,light,irradiance)
            yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
            matte=np.zeros((bh,bw),np.float32) if mask is None else mask[k,y0:y1,x0:x1].astype(np.float32)/255
            if mask is not None:
                hz=hand.depth(k,xx,yy)
                A*=1-matte*np.clip((hz-Z)/max(.04*s,1)+.5,0,1)
                # Soft contact shadow on the photographed palm, not a painted wall disc.
                shadow=np.exp(-((xx-p[0])/(.62*s))**2-((yy-p[1])/(.25*s))**2)
                contact=math.exp(-release*release*5-spread*spread*3)
                P[y0:y1,x0:x1]*=(1-.24*grow*fade*contact*shadow*matte)[...,None]
            # Preserve premultiplication while softening only subpixel mesh edges.
            C*=A[...,None]
            A=cv2.GaussianBlur(A,(0,0),.55)
            C=cv2.GaussianBlur(C,(0,0),.55)
            P[y0:y1,x0:x1]=P[y0:y1,x0:x1]*(1-A[...,None]*fade)+C*fade
            # Falling motes give the mass a direction; distant grains yield to the hand.
            for j,v in enumerate(dust):
                life=(age+j*.037)%1.2
                d=center+np.array([v[0]*s*(.3+life*.4),r*.5+s*(life*life*.65),v[2]*s*.35])
                ix,iy=int(d[0]),int(d[1])
                if not 0<=ix<bw or not 0<=iy<bh:
                    continue
                if mask is not None and matte[iy,ix]>.5 and d[2]<hand.depth(k,np.float32(ix+x0),np.float32(iy+y0)):
                    continue
                rr=max(1,round(dust_size[j]*s))
                opacity=.24*fade*grow*(1-life/1.2)
                ymin,ymax=max(0,iy-rr),min(bh,iy+rr+1);xmin,xmax=max(0,ix-rr),min(bw,ix+rr+1)
                dy,dx=np.mgrid[ymin:ymax,xmin:xmax]
                coverage=np.exp(-((dx-ix)**2+(dy-iy)**2)/(rr*rr))*opacity
                patch=P[y0+ymin:y0+ymax,x0+xmin:x0+xmax]
                patch[:]=patch*(1-coverage[...,None])+irradiance*meshes[j%13].colour*.7*coverage[...,None]
        out[k]=gam(P)
        if k%12==0:
            say("earth",k/n,f"Shaping stone: frame {k+1} of {n}")
    say("earth",1,"The earth is made")
    return out
