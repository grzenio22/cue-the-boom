"""Earth bending: tracked, rotating stone geometry with reflected light and falling grit.

The mesh is three dimensional; depth tests order its faces, orbiting fragments and
the photographed hand. Material detail lives in object coordinates, so it rotates
with the stone. This is art-directed CPU geometry, not a simulated fracture solver.
"""
from __future__ import annotations

import math
from itertools import combinations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .elements import Timeline
from .transfer import lin, gam
from .water import hand_of

FORMS = ("rock", "shatter", "volley", "lance")


@dataclass
class Stone:
    vertices: np.ndarray
    faces: np.ndarray
    phase: np.ndarray
    colour: np.ndarray
    origin: tuple = (0.,0.,0.)  # shared material coordinates across fracture cells


@lru_cache(maxsize=1)
def _stone_maps():
    root=Path(__file__).with_name('assets')/'stone'
    images=[]
    for name in ('diffuse','normal','ao'):
        image=cv2.imread(str(root/(name+'.jpg')))
        if image is None:
            raise RuntimeError('Bundled stone material is missing; reinstall CueSync')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(lin(image) if name=='diffuse' else image.astype(np.float32)/255)
    return images


def _stone_material(obj,normal):
    """Triplanar photographed albedo and normal detail, fixed in rock coordinates."""
    diffuse,normals,ao=_stone_maps();weights=np.abs(normal)**6;weights/=max(weights.sum(),1e-8)
    colour=np.zeros(obj.shape,np.float32);bump=np.zeros(obj.shape,np.float32)
    for axis,(u,v) in enumerate(((2,1),(0,2),(0,1))):
        weight=weights[axis]
        if weight<.005:
            continue
        x=((obj[...,u]*.43+.37)%1*diffuse.shape[1]).astype(np.float32)
        y=((obj[...,v]*.43+.21)%1*diffuse.shape[0]).astype(np.float32)
        def read(tex):return cv2.remap(tex,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        colour+=weight*read(diffuse)*(.45+.55*read(ao))
        mapped=read(normals)*2-1
        bump[...,u]+=weight*mapped[...,0]*.65
        bump[...,v]-=weight*mapped[...,1]*.65
    nrm=normal+bump;nrm/=np.maximum(np.linalg.norm(nrm,axis=-1,keepdims=True),1e-8)
    return colour,nrm


def _weather(vertices):
    """A continuous displacement shared by both sides of every fracture seam."""
    p=np.asarray(vertices)
    return p+.028*np.stack([np.sin(p[:,1]*13+p[:,2]*7),np.sin(p[:,2]*11-p[:,0]*9),
                           np.sin(p[:,0]*12+p[:,1]*8)],axis=-1)


def _clip(polygons, normal, offset):
    """Clip a closed convex polyhedron and seal its exposed fracture plane."""
    result=[];rim=[]
    for polygon in polygons:
        cut=[]
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            da,db=a@normal-offset,b@normal-offset
            if da<=1e-9:
                cut.append(a)
            if (da<0)!=(db<0):
                cross=a+(b-a)*da/(da-db);cut.append(cross);rim.append(cross)
        if len(cut)>=3:
            result.append(np.asarray(cut))
    if len(rim)>=3:
        rim=np.unique(np.round(rim,9),axis=0);center=rim.mean(0)
        axis=np.cross(normal,[1,0,0] if abs(normal[0])<.8 else [0,1,0]);axis/=np.linalg.norm(axis)
        across=np.cross(normal,axis)
        angle=np.arctan2((rim-center)@across,(rim-center)@axis)
        result.append(rim[np.argsort(angle)])
    return result


def fracture(seed):
    """Eight complementary solid cells, cut from one irregular parent boulder.

    Plane clipping keeps their silhouettes, interior faces and volume consistent
    between assembly and breakup, without a separate simulation dependency.
    """
    rng=np.random.default_rng(seed)
    phi=(1+math.sqrt(5))/2
    pts=np.array([(0,a,b*phi) for a in (-1,1) for b in (-1,1)]+
                 [(a,b*phi,0) for a in (-1,1) for b in (-1,1)]+
                 [(b*phi,0,a) for a in (-1,1) for b in (-1,1)],float)
    pts/=np.linalg.norm(pts,axis=1,keepdims=True)
    pts*=rng.uniform(.83,1.13,(12,1))*[1.12,.88,.96]
    polygons=[]
    for ids in combinations(range(12),3):
        a,b,c=pts[list(ids)];d=(pts-a)@np.cross(b-a,c-a)
        if np.all(d>=-1e-8) or np.all(d<=1e-8):
            polygons.append(pts[list(ids)])
    cells=[polygons]
    for axis in np.eye(3):
        normal=axis+rng.uniform(-.28,.28,3);normal/=np.linalg.norm(normal)
        offset=float(rng.uniform(-.13,.13))
        cells=[_clip(p,sign*normal,sign*offset) for p in cells for sign in (-1,1)]
    parts=[]
    phase=rng.uniform(0,6.28,5)
    for cell in cells:
        center=np.unique(np.concatenate(cell),axis=0).mean(0)
        vertices=[];faces=[]
        for polygon in cell:
            # Welding avoids duplicate intersection endpoints. A centre fan
            # preserves collinear boundary vertices shared with adjacent faces;
            # an arbitrary corner fan produced degenerate triangles at cuts.
            polygon=np.round(polygon,7)
            clean=[]
            for point in polygon:
                if not any(np.array_equal(point,v) for v in clean):
                    clean.append(point)
            if len(clean)<3:
                continue
            polygon=np.array(clean);middle=polygon.mean(0)
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                if np.linalg.norm(np.cross(a-middle,b-middle))<1e-10:
                    continue
                start=len(vertices);vertices.extend([middle-center,a-center,b-center])
                faces.append((start,start+1,start+2))
        # Midpoint tessellation breaks long silhouette edges while the same
        # displacement on every shared position keeps the fracture watertight.
        refined=[];points=[]
        for ids in faces:
            a,b,c=np.array(vertices)[list(ids)]+center
            ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
            for tri in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)):
                start=len(points);points.extend(tri);refined.append((start,start+1,start+2))
        pts=_weather(np.array(points));center=pts.mean(0)
        parts.append((Stone(pts-center,np.array(refined),phase.copy(),np.array([1.,.97,.92]),tuple(center)),center))
    return parts


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
    return Stone(pts, faces, rng.uniform(0, 6.28, 5), np.array([1., .97, .92]) * rng.uniform(.9, 1.1))


def _mineral_noise(position):
    """Smooth isotropic value noise in stone coordinates, stable as it turns."""
    cell = np.floor(position)
    f = position - cell
    f = f*f*f*(f*(f*6-15)+10)
    value = np.zeros(position.shape[:-1], np.float32)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                q = cell + np.array([x, y, z])
                h = np.sin(q @ np.array([127.1, 311.7, 74.7])) * 43758.5453
                weight = (f[...,0] if x else 1-f[...,0]) * (f[...,1] if y else 1-f[...,1]) * (f[...,2] if z else 1-f[...,2])
                value += (weight * (h-np.floor(h))).astype(np.float32)
    return value


def _mineral_albedo(obj, phase):
    """Granite-like mottling, quartz grains, mica and irregular fracture seams."""
    p = obj + phase[:3]
    broad = _mineral_noise(p*3.8)
    grain = _mineral_noise(p*24.0)
    fine = _mineral_noise(p*93.0)
    cracks = np.exp(-((broad-.48)/.017)**2) * np.clip((grain-.28)*2, 0, 1)
    quartz = np.clip((grain-.57)*4.8, 0, .75)
    mica = np.clip((.38-fine)*3.2, 0, .6)
    rock = np.clip(.64+.32*broad+.25*(fine-.5)-.34*cracks-.4*mica, .2, 1.1)
    strata=np.sin(p[...,1]*20+2.8*broad+1.5*np.sin(p[...,0]*3))
    seams=np.exp(-((strata+.86)/.14)**2)
    return (rock*(1-.22*seams))[...,None]*np.array([1.02,.98,.93])+quartz[...,None]*np.array([.25,.23,.20])


def rotation(t: float) -> np.ndarray:
    a, b = t*.57, t*.33+.4
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]) @ np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])


def raster(mesh: Stone, center, radius, turn, rgb, alpha, depth, light, irradiance, opacity=1.):
    """Opaque, textured mesh into a shared local depth buffer, nearest Z wins."""
    H, W = alpha.shape
    verts = mesh.vertices @ turn.T * radius + center
    inverse_turn=np.linalg.inv(turn)
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
        obj_normal=turn.T@normal;obj_normal/=max(np.linalg.norm(obj_normal),1e-8)
        material,material_normal=_stone_material(obj+np.asarray(mesh.origin),obj_normal)
        # Broad rounded normals and a smaller facet contribution preserve rugged
        # edges without making every triangle read as a low-poly game asset.
        smooth = obj @ turn.T
        smooth /= np.maximum(np.linalg.norm(smooth,axis=-1,keepdims=True),1e-8)
        material_normal=material_normal@inverse_turn
        material_normal/=np.maximum(np.linalg.norm(material_normal,axis=-1,keepdims=True),1e-8)
        facing = np.maximum(0, .12*(smooth@light)+.88*(material_normal@light))
        shade=.25+.75*facing
        colour=material*mesh.colour*irradiance*shade[...,None]
        rgb[y0:y1,x0:x1][covered]=colour[covered]
        alpha[y0:y1,x0:x1][covered]=opacity
        depth[y0:y1,x0:x1][covered]=z[covered]


def _events(tl, form):
    if form in ("shatter","volley","lance") and tl.hits:
        return [(int(f), min(tl.n,int(f)+15), float(z)) for f,z in zip(tl.hits,tl.sizes) if 0<=f<tl.n]
    active=np.asarray(tl.energy)>.25
    for a,b in tl.holds:
        active[max(0,a):min(tl.n,b)]=True
    edges=np.diff(np.r_[False,active,False].astype(int))
    return [(int(a),int(b),float(np.max(tl.energy[a:b],initial=.5)))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def choreography(track, tl, form):
    """Voice opens the performance; observed strokes select its physical action."""
    from .motion import analyze
    windows=_events(tl,form);result=[]
    for e in analyze(track,tl.camera).events:
        owners=[a for a,b,_ in windows if a+5<=e.frame<b+3]
        if not owners:
            continue
        result.append(dict(frame=e.frame,action={'swipe':'shard_fan','lift':'raise_stack','slam':'crumble'}[e.kind],
                           source='hand',direction=list(e.direction),strength=e.strength,event_start=max(owners)))
    if form!='rock':
        for a,b,g in windows:
            nearby=next((e for e in result if abs(e['frame']-(a+6))<=7),None)
            if nearby is None:
                angle=math.radians(tl.aim if tl.aim is not None else 0.)
                result.append(dict(frame=min(tl.n-1,a+6),action={'shatter':'crumble','volley':'shard_fan','lance':'lance'}[form],
                                   source='voice',direction=[math.sin(angle),-math.cos(angle)],strength=g,event_start=a))
            elif form in ('lance','volley'):
                nearby['action']='lance' if form=='lance' else 'shard_fan'
    return sorted(result,key=lambda e:e['frame'])


def _flight_center(hand, anchor, k, camera, *, falling=True):
    """Released mass inherits hand velocity once, then moves with the camera."""
    from .bending import camera_between

    s=float(hand.s[anchor]);p=hand.p[anchor]
    previous=max(0,anchor-1)
    previous_in_frame=camera_between(camera,previous,anchor) @ np.r_[hand.p[previous],1.]
    velocity=(p-previous_in_frame[:2])*24
    velocity*=min(1.,3*s/max(float(np.linalg.norm(velocity)),1e-8))
    age=(k-anchor)/24
    xy=p+np.array([.1*s*age,-.65*s+(10.0*s*age*age if falling else 0.)])+velocity*age
    transform=camera_between(camera,anchor,k)
    xy=(transform @ np.r_[xy,1.])[:2]
    zoom=math.sqrt(abs(float(np.linalg.det(transform[:2,:2]))))
    # Gravity releases the mass behind the open fingers instead of dissolving
    # a translucent disc into photographed skin.
    z = float(hand.depth(anchor,p[0],p[1],fingers=False)) + .22*s - (1.5*s*age if falling else 0.)
    return np.r_[xy,z],s*zoom


def _held(hand,k):
    s=float(hand.s[k]);p=hand.p[k]
    return np.r_[p+[0,-.68*s],float(hand.depth(k,*p,fingers=False))+.20*s]


def _projectiles(action,k,hand,parts,meshes,camera,gain):
    """Inertial launches evaluated in the birth frame, then moved with the room."""
    from .bending import camera_between
    born=action['frame'];age=(k-born)/24
    if age<0 or age>1.5:
        return []
    s=float(hand.s[born]);origin=_held(hand,born)
    u=np.asarray(action['direction']);side=np.array([-u[1],u[0]])
    transform=camera_between(camera,born,k)
    zoom=math.sqrt(abs(float(np.linalg.det(transform[:2,:2]))))
    kind=action['action'];objects=[]
    count=1 if kind=='lance' else (5 if kind=='shard_fan' else len(parts))
    for j in range(count):
        delay=j*.055 if kind=='shard_fan' else 0
        t=max(0,age-delay)
        if kind=='shard_fan':
            spread=(j-2)*.23
            direction=u*math.cos(spread)+side*math.sin(spread)
            off=side*(j-2)*.14*s+direction*s*(4.6+gain*2)*t+[0,2.5*s*t*t]
            mesh=meshes[j+1];radius=s*(.17+.025*(j%3))
            turn=rotation(j+t*5)
        elif kind=='lance':
            off=u*s*(6+3*gain)*t+[0,1.4*s*t*t]
            mesh=meshes[0];radius=.42*s
            direction=np.r_[u,0.];across=np.r_[side,0.]
            turn=np.column_stack([across,direction,[0,0,1]])@np.diag([.38,1.9,.45])
        else:
            mesh,offset=parts[j]
            direction=offset/max(np.linalg.norm(offset),.01)
            off=direction[:2]*s*(.48+2.2*t)+u*s*1.5*t+[0,4.8*s*t*t]
            radius=.50*s;turn=rotation(t*4+j*.12)
        position=origin.copy();position[:2]+=off
        position[:2]=(transform@np.r_[position[:2],1])[:2]
        position[2]-=s*(.65*t+(j%3)*.1)
        objects.append((mesh,position,radius*zoom,turn))
    return objects


def render(plate, track, mask, timeline: Timeline, form: str, seed=0, progress=None, *, coverage=None):
    if form not in FORMS:
        raise ValueError(f"earth form must be one of {FORMS}")
    n,h,w=plate.shape[:3]
    if timeline.n!=n or (mask is not None and mask.shape!=(n,h,w)):
        raise ValueError("plate, timeline and mask dimensions must agree")
    if coverage is not None:
        if coverage.shape!=(n,h,w) or coverage.dtype!=np.uint8:
            raise ValueError("coverage must be a uint8 array matching the plate")
        coverage.fill(0)
    events=_events(timeline,form);out=plate.copy()
    if not events:
        return out
    hand=hand_of(track,n,plate);parts=fracture(seed)
    meshes=[stone(seed+j*101) for j in range(7)]
    actions=choreography(track,timeline,form)
    rng=np.random.default_rng(seed+417)
    dust=rng.normal(size=(70,3));dust_size=rng.uniform(.005,.018,70)
    # Existing room light, including its colour and broad direction. No emission.
    env=lin(plate[::max(1,n//6),:max(1,h//2):8,::8])
    irradiance=np.clip(np.percentile(env,88,axis=(0,1,2)),.001,1)
    luma=env.mean((0,3));yy,xx=np.mgrid[:luma.shape[0],:luma.shape[1]]
    weight=np.maximum(luma-np.percentile(luma,45),0);total=max(weight.sum(),1e-6)
    lx=float((weight*(xx/max(1,luma.shape[1]-1)-.5)).sum()/total)
    light=np.array([lx*1.8-.18,-.72,.62]);light/=np.linalg.norm(light)
    say=progress or (lambda *args:None)
    for k in range(n):
        presence=float(hand.presence[k])
        objects=[];s=float(hand.s[k]);p=hand.p[k];contact=0.;dust_origin=None;dust_age=0.
        for a,b,gain in events:
            if not a<=k<min(n,b+38):
                continue
            age=(k-a)/24;release=max(0,(k-b)/24)
            fade=min(1.,max(0.,(1.5-release)/.18))
            if fade<=0:
                continue
            grow=float(np.clip((k-a+1)/8,0,1));grow=grow*grow*(3-2*grow)
            center=_held(hand,k);r=s*(.43+.08*gain)
            related=[v for v in actions if v['event_start']==a and v['frame']<=k]
            # A lift briefly opens the fracture into a stepped stack, then settles.
            lift=max([math.sin(min(1,(k-v['frame'])/22)*math.pi)*v['strength']
                      for v in related if v['action']=='raise_stack' and k-v['frame']<22]+[0.])
            destructive=next((v for v in related if v['action'] in ('crumble','lance') or (form=='volley' and v['action']=='shard_fan')),None)
            reformed=destructive is not None and form=='rock' and k-destructive['frame']>24
            body=destructive is None or reformed
            if reformed:
                grow=min(1,(k-destructive['frame']-24)/9)
            if release:
                center,size=_flight_center(hand,b-1,k,timeline.camera);r=size*(.43+.08*gain)
            turn=rotation(age*.65)
            body_presence=float(hand.presence[b-1]) if release else presence
            if body and body_presence>0:
                if form=='lance':
                    turn=turn@np.diag([.40,1.8,.48])
                for j,(mesh,off) in enumerate(parts):
                    # Whole cells travel into place; no sphere scaling out of nothing.
                    assembly=(1-grow)*(1.4+.17*j)
                    separation=1.10+.065*math.sin(age*2+j*.7)+.35*lift
                    local=off*r*separation
                    local[1]+=r*assembly*(1.2 if j%2 else -1.)
                    local[1]-=lift*s*(.3+.12*j)
                    at=center+turn@local
                    objects.append((mesh,at,r*max(.3,grow),turn,fade*body_presence))
                contact=max(contact,grow*fade*body_presence*math.exp(-release*release*5)*(1-.65*lift))
            if body_presence>0:
                dust_origin=center;dust_age=age
            for action in related:
                if action['action']=='raise_stack':
                    continue
                birth_presence=float(hand.presence[action['frame']])
                if birth_presence<=0:
                    continue
                for mesh,at,radius,spin in _projectiles(action,k,hand,parts,meshes,timeline.camera,gain):
                    objects.append((mesh,at,radius,spin,fade*birth_presence))
        if not objects:
            continue
        # Bound all actual transformed silhouettes, including long lances.
        bounds=[(c[:2],r*np.max(np.linalg.norm(mesh.vertices@turn.T,axis=1))+3)
                for mesh,c,r,turn,fade in objects]
        lo=np.min([c-r for c,r in bounds],axis=0);hi=np.max([c+r for c,r in bounds],axis=0)
        lo=np.minimum(lo,p-[s,s]);hi=np.maximum(hi,p+[s,s])
        x0,y0=np.maximum(np.floor(lo).astype(int),0);x1,y1=np.minimum(np.ceil(hi).astype(int)+1,[w,h])
        if x1<=x0 or y1<=y0:
            continue
        bh,bw=y1-y0,x1-x0;C=np.zeros((bh,bw,3),np.float32);A=np.zeros((bh,bw),np.float32);Z=np.full((bh,bw),-np.inf,np.float32)
        # Opaque geometry first, one shared depth buffer. Fade only the final lifecycle tail.
        fade=max(o[4] for o in objects)
        for mesh,center,r,turn,opacity in objects:
            raster(mesh,center-[x0,y0,0],r,turn,C,A,Z,light,irradiance,opacity)
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        matte=np.zeros((bh,bw),np.float32) if mask is None else mask[k,y0:y1,x0:x1].astype(np.float32)/255
        if mask is not None:
            hz=hand.depth(k,xx,yy)
            A*=1-matte*np.clip((hz-Z)/max(.04*s,1)+.5,0,1)
        P=lin(plate[k]);patch=P[y0:y1,x0:x1]
        shadow=np.exp(-((xx-p[0])/(.57*s))**2-((yy-p[1])/(.24*s))**2)
        patch*=(1-.28*contact*shadow*matte)[...,None]
        C*=A[...,None];A=cv2.GaussianBlur(A,(0,0),.65);C=cv2.GaussianBlur(C,(0,0),.65)
        if coverage is not None:
            coverage[k,y0:y1,x0:x1]=np.rint(A*255).astype(np.uint8)
        patch[:]=patch*(1-A[...,None])+C
        # Sparse falling dust reflects the room and yields to foreground fingers.
        if dust_origin is not None:
            for j,v in enumerate(dust):
                life=(dust_age+j*.037)%1.2
                d=dust_origin+np.array([v[0]*s*(.20+life*.35),s*(.22+life*life*.8),v[2]*s*.30])
                ix,iy=int(d[0])-x0,int(d[1])-y0
                if not 0<=ix<bw or not 0<=iy<bh:
                    continue
                if matte[iy,ix]>.5 and d[2]<hand.depth(k,np.float32(ix+x0),np.float32(iy+y0)):
                    continue
                rr=max(1,round(dust_size[j]*s));opacity=.20*fade*(1-life/1.2)
                ya,yb=max(0,iy-rr),min(bh,iy+rr+1);xa,xb=max(0,ix-rr),min(bw,ix+rr+1)
                dy,dx=np.mgrid[ya:yb,xa:xb];alpha=np.exp(-((dx-ix)**2+(dy-iy)**2)/(rr*rr))*opacity
                q=patch[ya:yb,xa:xb];q[:]=q*(1-alpha[...,None])+irradiance*np.array([.34,.28,.20])*alpha[...,None]
        out[k]=gam(P)
        if k%12==0:
            say("earth",k/n,f"Shaping stone: frame {k+1} of {n}")
    say("earth",1,"The earth is made")
    return out
