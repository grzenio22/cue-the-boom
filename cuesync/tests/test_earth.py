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
