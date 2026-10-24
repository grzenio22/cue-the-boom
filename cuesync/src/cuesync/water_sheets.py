"""Moving, open water surfaces and measured gesture choreography.

The main forms are continuous triangle sheets, not unions of spheres. Geometry,
surface flow and released motion have separate clocks: a still hand cannot freeze
the current, and a cast cannot teleport back to its hand. Depth is in the same
camera-facing pixel units as water.Hand. This is a single-view approximation, not
a reconstruction of the room or a fluid simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .motion import analyze

FPS = 24
NEW_FORMS = ("ribbon", "shield", "jet")


def smooth(a, b, x):
    t = np.clip((np.asarray(x) - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


@dataclass
class Surface:
    xyz: np.ndarray  # [along, across, xyz]; z grows towards the camera
    opacity: float = 1.
    foam: float = .4
    flow: float = 1.  # water travelling along the surface, hand lengths/second
    phase: float = 0.
    released: bool = False


def strip(points, width, time, size, *, opacity=1., foam=.4, flow=1., phase=0., twist=.4):
    """A continuous ribbon with rolled, travelling folds and an uneven free edge."""
    points = np.asarray(points, np.float32)
    u = np.linspace(0, 1, len(points), dtype=np.float32)
    v = np.linspace(-1, 1, 9, dtype=np.float32)
    tangent = np.gradient(points, axis=0)
    across = np.c_[-tangent[:, 1], tangent[:, 0], np.zeros(len(u))]
    across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-5)
    angle = twist * np.sin(u * 7. - time * 2.1 + phase)
    across *= np.cos(angle)[:, None]
    across[:, 2] = np.sin(angle)
    width = np.broadcast_to(np.asarray(width, np.float32), u.shape)
    edge = 1 + .13*np.sin(19*u-time*7+phase) + .055*np.sin(39*u-time*11)
    xyz = points[:, None] + across[:, None] * (width*edge)[:, None, None] * v[None, :, None]
    # Long crests move through the sheet; the edge rolls over instead of a flat decal.
    folds = (.10*np.sin(13*u[:, None]-time*8+2*v[None]+phase)
             + .035*np.sin(29*u[:, None]-time*13-3*v[None]))
    xyz[:, :, 2] += size * folds * (1-.25*v[None]**2)
    xyz[:, :, 2] += width[:, None] * .6 * v[None]**2
    return Surface(xyz.astype(np.float32), float(opacity), foam, flow, phase)


def held_surfaces(form, centre, radius, size, time, energy, phase=0.):
    """Different silhouettes: rolling cup, breaking curl, spiral, curtain, jet."""
    c = np.asarray(centre, float)
    u = np.linspace(0, 1, 64)
    r = radius
    surfaces = []
    if form in ("orb", "surge", "ribbon"):
        turns = 4.65 if form == "orb" else (4.05 if form == "surge" else 8.8)
        a = .45 + u*turns + .22*math.sin(time*1.8+phase)
        rr = r*(.86+.12*np.sin(12*u-time*5)+.07*np.sin(23*u-time*9))
        if form == "ribbon":
            rr *= .8+.35*u
        pts = np.c_[c[0]+rr*np.cos(a), c[1]+rr*.66*np.sin(a), c[2]+rr*.75*np.sin(a+.6)]
        if form == "ribbon":
            pts[:, 1] += r*(.7-1.8*u)
        widths = r*(.46 if form == "orb" else .53 if form == "surge" else .30)
        widths *= (.45+.55*np.sin(np.pi*u)**.5)*(1-.7*smooth(.85,1,u))
        surfaces.append(strip(pts, widths, time, size, foam=.66 if form=="ribbon" else .85,
                              phase=phase, twist=.8))
        if form != "ribbon":
            a = 2.5 + 3.25*u + .25*math.sin(time*2.5)
            pts = np.c_[c[0]+r*.70*np.cos(a), c[1]+r*(.20+.55*np.sin(a)),
                        c[2]-r*.45+r*.4*np.sin(a)]
            surfaces.append(strip(pts, r*.23*np.sin(np.pi*u)**.5, time+.6, size,
                                  foam=.38, phase=phase+2, twist=.7))
    elif form == "shield":
        # Curved vertical curtain: broad translucent centre, crest rolling forward.
        v = np.linspace(0, 1, 13)
        xx = (u-.5)*size*1.9
        arch=np.sin(np.pi*u)**.55
        top = c[1]-size*(.18+.65*arch+.09*np.sin(17*u-time*4))
        bottom = c[1]+size*(.30+.12*np.sin(15*u-time*3))
        xyz = np.empty((len(u),len(v),3),np.float32)
        xyz[:,:,0] = c[0]+xx[:,None]*(1-.65*v[None]**2) + size*.035*np.sin(15*u[:,None]-time*4+v*5)
        xyz[:,:,1] = top[:,None]*(1-v)+bottom[:,None]*v
        xyz[:,:,2] = (c[2]+size*(.30*np.cos((u-.5)*np.pi))[:,None]
                        +size*(.16*np.sin(13*u[:,None]+v*5-time*5)
                               +.20*np.exp(-v[None]*7)*np.cos(v[None]*8-time*3)))
        surfaces.append(Surface(xyz,1.,.72,1.3,phase))
    elif form == "jet":
        for j in range(2):
            pts = np.c_[c[0]+size*(.11*np.sin(u*10-time*5+j)+j*.07),
                        c[1]+size*(.40-2.0*u),
                        c[2]+size*(.12*np.sin(u*8-time*3+j))]
            widths = size*(.10-.055*u)*(.65+.35*np.sin(18*u-time*10+j)**2)
            surfaces.append(strip(pts,widths,time,size,foam=.58,flow=2.,phase=phase+j,twist=.7))
    return surfaces


def choreography(track, tl, form):
    """Only observed, voiced motion makes actions; hits also work on a still hand."""
    motion = analyze(track, tl.camera)
    actions = []
    for event in motion.events:
        k = event.frame
        if not tl.held_at(k):
            continue
        action = {"swipe":"water_blade", "lift":"rising_wave", "slam":"splash_crown"}[event.kind]
        actions.append(dict(frame=k, action=action, direction=list(event.direction),
                            strength=round(event.strength,3), source="hand"))
    # A held shhh controls the body. A separate sploosh propels a stream; one
    # accent is enough when a measured stroke and voice happen together.
    if form in NEW_FORMS:
        for k, strength in zip(tl.hits,tl.sizes):
            # Voice does not infer a stroke from motion, so a stable observed
            # hand may launch below the stricter kinematic confidence threshold.
            observed=(0<=k<len(motion.valid) and track.get('src',['mp']*tl.n)[k] not in ('lost','hold')
                      and track.get('conf',[1.]*tl.n)[k]>=.55 and np.isfinite(track['xy'][k]).all())
            if observed and all(abs(k-a['frame'])>5 for a in actions):
                actions.append(dict(frame=int(k),action="pulse_jet",direction=None,
                                    strength=float(strength),source="voice"))
    return sorted(actions,key=lambda a:a['frame'])


def action_surfaces(action, k, hand, tl, seed):
    """Finite ballistic shapes in their birth frame, then transported by camera."""
    from .bending import camera_between
    from .water import _aim
    birth = action['frame']; age = (k-birth)/FPS
    if age<0 or age>=1.25:
        return []
    s = float(hand.s[birth]); p = hand.p[birth]
    z = float(hand.depth(birth,np.array([p[0]]),np.array([p[1]]))[0])+.35*s
    origin = np.array([p[0],p[1]-.35*s,z])
    d = np.asarray(action['direction'] if action['direction'] is not None else _aim(tl,hand,birth))
    d /= max(np.linalg.norm(d),1e-6)
    across = np.array([-d[1],d[0]])
    gain = float(action['strength']); fade = float(smooth(0,.10,age)*(1-smooth(.8,1.25,age)))
    u = np.linspace(0,1,56); phase = seed*.71+birth*.33
    kind = action['action']
    if kind == "splash_crown":
        a = np.linspace(0,2*np.pi,80)
        rr = s*(.18+age*1.45)
        tips = np.maximum(0,np.sin(9*a+phase))**4
        pts = np.c_[origin[0]+rr*np.cos(a), origin[1]+rr*.34*np.sin(a)-s*(.48*np.sin(np.pi*min(age,1)))*(1+tips),
                    origin[2]+rr*.75*np.sin(a)]
        surfaces=[strip(pts,s*(.07+.10*tips)*(1-.6*age),age,s,opacity=fade,foam=.75,phase=phase)]
    elif kind == "rising_wave":
        c = origin + np.r_[d*s*age*.7,0.]
        surfaces=held_surfaces('shield',c,s*.6,s,age*1.8+phase,gain,phase)
        for surface in surfaces:
            surface.opacity=fade
            surface.xyz[:,:,1]-=s*(.3+age)*gain
    else:
        distance = age*s*(3.2+gain)
        extent = s*(.5+min(age,.4)*2.4)
        trail = distance-extent*(1-u)
        pts = np.zeros((len(u),3));pts[:]=origin
        pts[:,:2]+=trail[:,None]*d
        pts[:,:2]+=across*(s*.16*np.sin(np.pi*u))[:,None]
        pts[:,1]+=s*age*age*.5
        pts[:,2]+=s*.16*np.sin(u*6-age*7)
        width = s*(.22 if kind=='water_blade' else .11)*np.sin(np.pi*u)**.5
        surfaces=[strip(pts,width,age,s,opacity=fade,foam=.60,flow=2.4,phase=phase)]
    matrix = camera_between(tl.camera,birth,k)
    scale = math.sqrt(abs(np.linalg.det(matrix[:2,:2])))
    for surface in surfaces:
        surface.released=True
        shape=surface.xyz.shape
        points=surface.xyz.reshape(-1,3)
        points[:,:2]=(np.c_[points[:,:2],np.ones(len(points))]@matrix.T)[:,:2]
        points[:,2]*=scale
        surface.xyz=points.reshape(shape)
    return surfaces


def apply_scenes(scenes, form, tl, track, hand, seed):
    """Replace held glass volumes while retaining the existing wet-palm lifecycle."""
    from .water import orb_plan, Drops
    plan=orb_plan(tl,track)
    actions=choreography(track,tl,form)
    spray=Drops(tl.camera,(int(track['H']),int(track['W'])),float(np.median(hand.s)))
    released=Drops(tl.camera,(int(track['H']),int(track['W'])),float(np.median(hand.s)))
    rng=np.random.default_rng(seed+915)
    for k,sc in enumerate(scenes):
        spray.step()
        released.step()
        if form in NEW_FORMS:
            # These forms continue through repeated gestures and voice hits.
            # A five-frame tail drains after each voiced interval.
            intervals=[(a,b) for a,b in tl.holds if a<=k<b+5]
            sc.blobs=[];sc.ribbon=[];sc.held=None;sc.film=[];sc.cast=[];sc.coat=None
            sc.wet=[];sc.band=[];sc.plops=[]
            if intervals:
                a,b=intervals[-1];s=float(hand.s[k]);p=hand.p[k]
                visibility=float(smooth(a,a+5,k)*(1-smooth(b,b+5,k)))
                z=float(hand.depth(k,np.array([p[0]]),np.array([p[1]]))[0])+.5*s
                strength=.85+.25*float(np.sqrt(np.mean(tl.energy[max(0,k-4):k+1])))
                c=np.array([p[0],p[1]-.55*s,z]);r=.48*s*strength
                sc.sheets=held_surfaces(form,c,r,s,k/FPS,float(tl.energy[k]),seed*.41)
                for surface in sc.sheets:
                    surface.opacity*=visibility
            wet_intervals=[(a,b) for a,b in tl.holds if a<=k<b+20]
            if wet_intervals:
                a,b=wet_intervals[-1];s=float(hand.s[k]);p=hand.p[k]
                wet=float(smooth(a,a+6,k)*(1-smooth(b,b+20,k)))
                sc.wet=[(p[0],p[1],.32*s,.20*s,0.,.10*wet,seed*.4,.05*wet,.4)]
        elif form in ('orb','surge') and sc.held is not None and (plan.release is None or k<plan.release):
            x,y,z,r,weight=sc.held;s=float(hand.s[k])
            if r>.08*s:
                sc.sheets=held_surfaces(form,(x,y,z),r*1.22,s,k/FPS,float(tl.energy[k]),seed*.41)
                visibility=float(smooth(plan.formed-5,plan.formed+2,k))
                for surface in sc.sheets:
                    surface.opacity*=visibility
                # Keep only small free spray: no closed solid core under the sheets.
                sc.blobs=[b for b in sc.blobs if b.free and b.r<.08*s]
                sc.ribbon=[];sc.sparks=[];sc.bubbles=[];sc.trains=[];sc.sweep=None
        elif (form in ('orb','surge') and plan.throw and plan.release is not None
              and k>=plan.release and hand.presence[plan.release]>.5 and sc.coat is None):
            # Carry the released reservoir's measured trajectory into a breaking
            # sheet as well. Do not turn the new current back into a glass marble.
            s=float(hand.s[k])
            bodies=[b for b in sc.blobs if not b.free and not b.stream and b.r>.15*s]
            for b in bodies:
                pieces=held_surfaces('surge',(b.x,b.y,b.z),b.r,s,k/FPS,float(tl.energy[k]),seed*.41)
                for surface in pieces:
                    surface.released=True;surface.opacity*=b.w
                sc.sheets.extend(pieces)
            sc.blobs=[b for b in sc.blobs if b not in bodies]
        for action in actions:
            sc.sheets.extend(action_surfaces(action,k,hand,tl,seed))
        # Break the free edges into real, independently falling droplets. Their
        # birth positions and velocities are never moved with the later hand.
        if k%2==0:
            for surface in sc.sheets:
                if surface.opacity<.3 or (not surface.released and hand.presence[k]<=.5):continue
                a=len(surface.xyz);j=int(rng.integers(max(1,a//2),a-1))
                edge=surface.xyz[j,-1 if k%4 else 0]
                tangent=surface.xyz[min(a-1,j+1),4]-surface.xyz[max(0,j-1),4]
                tangent/=max(np.linalg.norm(tangent),1e-6)
                s=float(hand.s[k])
                for _ in range(3):
                    velocity=tangent*s*.07+rng.normal(0,.018*s,3)
                    pool=released if surface.released else spray
                    pool.add(k,edge+rng.normal(0,.012*s,3),velocity,s*rng.uniform(.004,.012),
                             foam=.42,life=16,fade=5,g=3*s/FPS**2)
        sc.blobs.extend(spray.blobs(k))
        sc.released_spray=released.blobs(k)
        sc.blobs.extend(sc.released_spray)
    return scenes


def raster(surface, shape):
    """Depth-tested smooth normals and material coordinates, limited to its box."""
    H,W=shape;xyz=surface.xyz;points=xyz.reshape(-1,3)
    x0=max(0,int(np.floor(points[:,0].min()))-2);x1=min(W,int(np.ceil(points[:,0].max()))+3)
    y0=max(0,int(np.floor(points[:,1].min()))-2);y1=min(H,int(np.ceil(points[:,1].max()))+3)
    if x1<=x0 or y1<=y0:
        return None
    a,b=xyz.shape[:2]
    normals=np.cross(np.gradient(xyz,axis=0),np.gradient(xyz,axis=1))
    normals/=np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-6)
    normals*=np.where(normals[:,:,2:3]<0,-1,1)
    uv=np.stack(np.meshgrid(np.linspace(0,1,a),np.linspace(-1,1,b),indexing='ij'),axis=2)
    attributes=np.concatenate([xyz[:,:,2:3],normals,uv],axis=2).reshape(-1,6)
    grid=np.arange(a*b).reshape(a,b)
    faces=np.concatenate([np.stack([grid[:-1,:-1],grid[1:,:-1],grid[1:,1:]],axis=-1).reshape(-1,3),
                          np.stack([grid[:-1,:-1],grid[1:,1:],grid[:-1,1:]],axis=-1).reshape(-1,3)])
    values=np.zeros((y1-y0,x1-x0,6),np.float32);values[:,:,0]=-1e10
    cover=np.zeros(values.shape[:2],np.float32)
    for ids in faces:
        tri=points[ids,:2]-[x0,y0]
        lo=np.maximum(0,np.floor(tri.min(0)).astype(int));hi=np.minimum([x1-x0,y1-y0],np.ceil(tri.max(0)).astype(int)+1)
        if np.any(hi<=lo):continue
        x,y=np.meshgrid(np.arange(lo[0],hi[0],dtype=np.float32),np.arange(lo[1],hi[1],dtype=np.float32))
        p,q,r=tri;det=(q[1]-r[1])*(p[0]-r[0])+(r[0]-q[0])*(p[1]-r[1])
        if abs(det)<1e-6:continue
        wa=((q[1]-r[1])*(x-r[0])+(r[0]-q[0])*(y-r[1]))/det
        wb=((r[1]-p[1])*(x-r[0])+(p[0]-r[0])*(y-r[1]))/det
        wc=1-wa-wb
        inside=(wa>=-1e-5)&(wb>=-1e-5)&(wc>=-1e-5)
        vals=wa[:,:,None]*attributes[ids[0]]+wb[:,:,None]*attributes[ids[1]]+wc[:,:,None]*attributes[ids[2]]
        patch=values[lo[1]:hi[1],lo[0]:hi[0]];take=inside&(vals[:,:,0]>patch[:,:,0])
        patch[take]=vals[take];cover[lo[1]:hi[1],lo[0]:hi[0]][take]=1.
    # Pixel-size edge coverage; attributes normalized over only valid neighbours.
    weight=cv2.GaussianBlur(cover,(0,0),.60)
    valid=cover>.5
    for j in range(6):
        source=np.where(valid,values[:,:,j],0)
        values[:,:,j]=cv2.GaussianBlur(source,(0,0),.60)/np.maximum(weight,1e-6)
    return (y0,y1,x0,x1),values,weight


def paint(P, surfaces, k, hand, room, noise, matte):
    """Local refraction, moving froth and scene-lit crests on explicit sheets."""
    from .water import _remap
    from .fx_light import _LUMA
    boxes=[];s=max(float(hand.s[k]),1.);time=k/FPS
    # Back to front. Each surface keeps its own z-buffer across folds.
    for surface in sorted(surfaces,key=lambda sf:float(np.mean(sf.xyz[:,:,2]))):
        if surface.opacity<.001:continue
        result=raster(surface,P.shape[:2])
        if result is None:continue
        box,values,cover=result;y0,y1,x0,x1=box
        iy,ix=np.nonzero(cover>.001)
        if not len(ix):continue
        x=(ix+x0).astype(np.float32);y=(iy+y0).astype(np.float32)
        z,nx,ny,nz,u,v=values[iy,ix].T
        uv_phase=u*14-time*surface.flow*4+surface.phase
        ripple=np.sin(uv_phase+1.8*v)
        detail=noise.along(18,u*s*2,v,k,.12*s,.30,.04*s*surface.flow)
        grain=noise.along(19,u*s*2,v,k,.025*s,.12,.04*s*surface.flow)
        normal=np.c_[nx+.16*np.cos(uv_phase+v*3),ny+.12*np.sin(uv_phase*1.2-v*4),nz]
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-5)
        nx,ny,nz=normal.T;nz=np.abs(nz)
        thickness=.035*s*(.65+.35*(1-v*v))
        # The background is not known in 3D. Stay local, even at grazing angles.
        offset=np.c_[nx,ny]*thickness[:,None]*(1.2+.5*ripple[:,None])
        sampled=_remap(P,x+offset[:,0],y+offset[:,1],cv2.BORDER_REPLICATE)
        absorb=np.exp(-np.array([.10,.035,.018])*thickness[:,None]/s)
        base=sampled*absorb
        reflection=room.reflect(k,2*nx*nz,2*ny*nz)
        fresnel=.025+.58*(1-nz)**4
        base=base*(1-fresnel[:,None])+reflection*fresnel[:,None]
        light=room.lit(k)
        key=room.key(k,(float(x.mean()),float(y.mean())))
        half=key+np.array([0,0,1.]);half/=np.linalg.norm(half)
        facing=np.maximum(0,normal@half)
        # Small, broken highlights can mirror a source brighter than diffuse foam.
        spec=facing**48*smooth(-.1,1.0,detail)*(.35+.65*smooth(-.5,.7,grain))
        crest=smooth(.68,.97,ripple)*smooth(-.5,.6,detail)
        edge=smooth(.64,.96,np.abs(v))
        foam=np.clip(surface.foam*(.10+.85*crest+.95*edge*smooth(-.25,.65,detail)
                                  +.32*smooth(.1,.8,detail))
                     *(.20+.80*smooth(-.5,.6,grain)),0,.85)
        # Froth scatters the measured room illumination; it is never emissive.
        base=base*(1-foam[:,None])+light*(.75+.25*nz[:,None])*foam[:,None]
        source=light/(1+light*2.)
        base+=source*(2.8*spec+.45*crest)[:,None]
        # A rolling edge shades itself, interrupted by travelling crests.
        base*=1-(.15*edge*(1-crest))[:,None]
        alpha=cover[iy,ix]*surface.opacity
        # Thin ends open into holes and wisps, never a solid plastic cutout.
        thin=np.maximum(smooth(.72,.99,np.abs(v)),smooth(.88,1.,u))
        alpha*=1-thin*smooth(.25,.9,-detail)*.85
        if matte is not None:
            front=smooth(-.04*s,.04*s,z-hand.depth(k,x,y))
            alpha*=1-matte[iy+y0,ix+x0].astype(float)/255*(1-front)
        patch=P[y0:y1,x0:x1]
        patch[iy,ix]=patch[iy,ix]*(1-alpha[:,None])+np.clip(base,0,1)*alpha[:,None]
        boxes.append(box)
    return boxes
