"""Solid voice-driven earth must have mass, depth, tracking and a finite release."""
import importlib

import cv2
import numpy as np
import pytest

from cuesync.elements import Timeline


def scene(n=40):
    y, x = np.mgrid[:128, :96]
    image = np.stack([130 + 8*np.sin(x/5), 123 + 7*np.sin(y/7), 110 + 5*np.cos(x/4)], -1).astype(np.uint8)
    plate = np.repeat(image[None], n, 0)
    track = dict(W=96, H=128, fps=24, xy=[[48., 90.]]*n, dir=[[0., -1.]]*n,
                 scale=[26.]*n, src=["mp"]*n, conf=[1.]*n, speed=[0.]*n,
                 landmarks=[None]*n, gesture=["Open_Palm"]*n, events=[])
    mask = np.zeros((n,128,96), np.uint8)
    for m in mask:
        cv2.ellipse(m,(48,94),(20,11),0,0,360,255,-1)
    e=np.zeros(n);e[6:23]=.8
    return plate, track, mask, Timeline(n,e,[6],[1.],[(6,23)],tap_frame=6)


def renderer():
    assert importlib.util.find_spec("cuesync.earth") is not None, "The agreed earth element is missing"
    return importlib.import_module("cuesync.earth")


def test_material_coverage_excludes_background_and_the_contact_shadow():
    p, t, m, tl = scene()
    coverage = np.full(p.shape[:3], 255, np.uint8)
    result = renderer().render(p, t, m, tl, "rock", coverage=coverage)
    assert not coverage[:6].any()
    assert not coverage[:, :10].any()
    assert (coverage > 250).sum() > 300
    # A material replacement must leave the photographed palm's shadow alone.
    changed = np.max(np.abs(result.astype(float) - p), axis=-1) > 4
    assert (changed & (m > 0) & (coverage == 0)).sum() > 20


@pytest.mark.parametrize("form", ["rock", "shatter"])
def test_earth_is_solid_local_deterministic_and_does_not_change_pre_cue_frames(form):
    p,t,m,tl=scene()
    earth=renderer()
    out=earth.render(p,t,m,tl,form,seed=5)
    again=earth.render(p,t,m,tl,form,seed=5)
    assert out.shape==p.shape and out.dtype==np.uint8
    assert np.array_equal(out,again)
    assert np.array_equal(out[:6],p[:6])
    assert np.array_equal(out[:,:15],p[:,:15])
    assert np.count_nonzero(np.abs(out[15].astype(int)-p[15])>20)>60
    assert np.array_equal(p,scene()[0]), "Never mutate the source plate"


def test_earth_is_silent_when_the_performance_is_silent():
    p,t,m,tl=scene()
    silent=Timeline(tl.n,np.zeros(tl.n),[],[],[])
    assert np.array_equal(renderer().render(p,t,m,silent,"rock"),p)


def test_earth_releases_and_settles_instead_of_staying_attached_forever():
    p,t,m,tl=scene(130)
    out=renderer().render(p,t,m,tl,"rock",seed=8)
    assert np.array_equal(out[-4:],p[-4:])


def test_earth_does_not_emit_light_into_an_unlit_room():
    p,t,m,tl=scene()
    p[:]=8
    out=renderer().render(p,t,m,tl,"rock",seed=4)
    assert out.max()<=35, "Stone reflects the room; it is not an emissive magical light"


def test_bad_form_timeline_and_mask_are_rejected():
    p,t,m,tl=scene()
    earth=renderer()
    with pytest.raises(ValueError): earth.render(p,t,m,tl,"lava")
    with pytest.raises(ValueError): earth.render(p,t,m,Timeline(2,[1,1],[],[]),"rock")
    with pytest.raises(ValueError): earth.render(p,t,m[:2],tl,"rock")


def test_stone_depth_order_does_not_depend_on_draw_order():
    earth=renderer()
    light=np.array([-.45,-.72,.62]);light/=np.linalg.norm(light)
    mesh=earth.stone(5)
    def draw(order):
        rgb=np.zeros((80,80,3),np.float32);alpha=np.zeros((80,80),np.float32);depth=np.full((80,80),-np.inf,np.float32)
        for z in order:
            earth.raster(mesh,np.array([40,40,z]),22,earth.rotation(z*.03),rgb,alpha,depth,light,np.ones(3))
        return rgb,alpha,depth
    front_first=draw([40,-40]);back_first=draw([-40,40])
    assert all(np.array_equal(a,b) for a,b in zip(front_first,back_first))
    assert front_first[2][40,40]>40


def test_earth_never_attaches_to_a_missing_hand():
    p,t,m,tl=scene()
    t['src']=['lost']*tl.n
    assert np.array_equal(renderer().render(p,t,m,tl,'rock'),p)


def test_released_stone_follows_camera_but_not_subsequent_hand_motion():
    p,t,m,tl=scene()
    hand=renderer().hand_of(t,tl.n,p)
    base,size=renderer()._flight_center(hand,22,30,None)
    cam=np.repeat(np.eye(3)[None],tl.n,axis=0);cam[23:,0,2]=18
    moved,other_size=renderer()._flight_center(hand,22,30,cam)
    assert np.allclose(moved-base,[18,0,0]) and other_size==size
    hand.p[23:,0]+=50
    independent,_=renderer()._flight_center(hand,22,30,None)
    assert np.allclose(independent,base)


def test_released_stone_falls_behind_the_hand_with_a_solid_surface():
    p,t,m,tl=scene(65)
    earth=renderer();hand=earth.hand_of(t,tl.n,p)
    initial,size=earth._flight_center(hand,22,22,None)
    fallen,_=earth._flight_center(hand,22,34,None)
    assert fallen[1]-initial[1] > 2*size
    assert fallen[2] < hand.depth(34,fallen[0],fallen[1],fingers=False)


@pytest.mark.parametrize('seed',[0,17,18,91])
def test_fracture_cells_are_closed_solids_with_shared_boundary_vertices(seed):
    parts=renderer().fracture(seed)
    assert 6<=len(parts)<=12
    vertices=np.concatenate([m.vertices+c for m,c in parts])
    def volume(mesh):
        v=mesh.vertices; total=0.
        for face in mesh.faces:
            a,b,c=v[face];normal=np.cross(b-a,c-a)
            total+=abs(float(a@normal))/6
        return total
    pieces=sum(volume(m) for m,_ in parts)
    # A finite solid volume, with no collapsed cells or giant interior faces.
    assert 1.5<pieces<4.5
    # Each closed cell has every geometric edge exactly twice. Neighbouring
    # cells retain shared boundary positions after surface weathering.
    from collections import Counter
    sets=[]
    for mesh,center in parts:
        verts=[tuple(v) for v in np.round(mesh.vertices+center,7)]
        edges=Counter(tuple(sorted((verts[a],verts[b]))) for f in mesh.faces
                      for a,b in zip(f,np.roll(f,-1)))
        assert set(edges.values())=={2}
        sets.append(set(verts))
    neighbours=[sum(len(a&b)>=3 for j,b in enumerate(sets) if i!=j) for i,a in enumerate(sets)]
    assert min(neighbours)>=3
    assert max(np.linalg.norm(c) for _,c in parts)>.4


def test_voice_gated_motion_drives_distinct_earth_actions():
    p,t,m,tl=scene(60)
    xy=np.asarray(t['xy']);xy[12:23,0]+=np.linspace(0,45,11);xy[23:,0]+=45;t['xy']=xy.tolist()
    plan=renderer().choreography(t,tl,'rock')
    assert any(e['action']=='shard_fan' for e in plan)
    silent=Timeline(tl.n,np.zeros(tl.n),[],[],[])
    assert renderer().choreography(t,silent,'rock')==[]
    t['src']=['lost']*tl.n
    assert not any(e['source']=='hand' for e in renderer().choreography(t,tl,'rock'))


@pytest.mark.parametrize('form',['volley','lance'])
def test_new_projectiles_are_deterministic_and_keep_voice_timing(form):
    p,t,m,tl=scene(45)
    result=renderer().render(p,t,m,tl,form,seed=17)
    assert np.array_equal(result[:6],p[:6])
    assert not np.array_equal(result[12:25],p[12:25])
    assert np.array_equal(result,renderer().render(p,t,m,tl,form,seed=17))


def test_lance_and_volley_have_different_silhouettes_and_release_paths():
    p,t,m,tl=scene(45)
    a=renderer().render(p,t,m,tl,'lance',seed=17)
    b=renderer().render(p,t,m,tl,'volley',seed=17)
    assert np.count_nonzero(np.max(abs(a.astype(float)-b),axis=-1)>10)>500


@pytest.mark.parametrize('kind',['lance','shard_fan','crumble'])
def test_new_projectiles_keep_birth_momentum_and_follow_the_camera(kind):
    earth=renderer();p,t,m,tl=scene(60);hand=earth.hand_of(t,tl.n,p)
    parts=earth.fracture(17);meshes=[earth.stone(j) for j in range(7)]
    action=dict(frame=16,action=kind,direction=[1.,0.])
    def positions(cam):
        return np.array([o[1] for o in earth._projectiles(action,25,hand,parts,meshes,cam,.8)])
    base=positions(None)
    hand.p[17:,0]+=70
    assert np.array_equal(base,positions(None))
    cam=np.repeat(np.eye(3)[None],tl.n,0);cam[17:,0,2]=13
    assert np.allclose(positions(cam)-base,[13,0,0])


def test_close_voice_hits_do_not_assign_the_same_hand_action_to_two_events():
    p,t,m,tl=scene(60)
    xy=np.asarray(t['xy']);xy[12:23,0]+=np.linspace(0,45,11);xy[23:,0]+=45;t['xy']=xy.tolist()
    tl.hits=[6,10];tl.sizes=[1.,1.]
    plan=renderer().choreography(t,tl,'volley')
    hand_actions=[e for e in plan if e['source']=='hand']
    assert len(hand_actions)==1 and hand_actions[0]['action']=='shard_fan'
    assert hand_actions[0]['event_start'] in tl.hits


@pytest.mark.parametrize('form',['volley','lance'])
def test_launched_projectiles_survive_the_hand_leaving_the_picture(form,monkeypatch):
    p,t,m,tl=scene(25);earth=renderer();hand=earth.hand_of(t,tl.n,p)
    hand.presence[15:]=0
    monkeypatch.setattr(earth,'hand_of',lambda *args:hand)
    m[15:]=0
    coverage=np.zeros(p.shape[:3],np.uint8)
    out=earth.render(p,t,m,tl,form,seed=17,coverage=coverage)
    assert coverage[16:19].max()>200
    assert not np.array_equal(out[16:19],p[16:19])


def test_scanned_surface_and_new_fractures_have_distinct_material_coordinates():
    earth=renderer();parts=earth.fracture(17)
    for mesh,offset in parts:
        assert mesh.exterior.any() and (~mesh.exterior).any()
        positions=mesh.vertices[mesh.faces[mesh.exterior]].mean(1)+offset
        colour=earth._scan_sample('albedo',positions)
        normals=earth._scan_sample('normal',positions)
        assert np.isfinite(colour).all() and np.isfinite(normals).all()
        assert 0<=colour.min()<colour.max()<=1
        assert np.linalg.norm(normals,axis=1).min()>.7
    # Sampling on either side of the longitude seam must stay continuous.
    edge=np.array([[-1.,0.,1e-6],[-1.,0.,-1e-6]])
    assert np.max(abs(np.diff(earth._scan_sample('albedo',edge),axis=0)))<.02
