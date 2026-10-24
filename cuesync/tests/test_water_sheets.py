"""Surface/material regressions and the actual rules of the water playground."""
import numpy as np
import pytest

from cuesync import water, water_sheets as sheets
from cuesync.elements import Timeline, FORMS
from cuesync.transfer import lin


def track(n=72):
    return dict(W=160,H=240,xy=[[80.,160.]]*n,scale=[50.]*n,
                dir=[[0.,-1.]]*n,conf=[1.]*n,src=['mp']*n,landmarks=[None]*n)


def timeline(n=72,holds=((4,65),),hits=()):
    return Timeline(n,np.full(n,.6),list(hits),[.8]*len(hits),list(holds))


def test_dark_room_ignores_the_bright_skin_and_has_no_emissive_floor():
    plate=np.full((2,240,160,3),24,np.uint8);mask=np.zeros(plate.shape[:3],np.uint8)
    plate[:,100:230,40:145]=[210,104,61];mask[:,100:230,40:145]=255
    room=water.Room(plate,mask)
    assert room.lit(0).max()<.04
    assert room.lit(0).min()/room.lit(0).max()>.9
    assert water.Room(np.zeros_like(plate)).lit(0).max()==0


@pytest.mark.parametrize('form',('orb','surge',*sheets.NEW_FORMS))
def test_still_holds_have_travelling_geometry_and_no_solid_core(form):
    t=track();tl=timeline();hand=water.hand_of(t,tl.n)
    scenes=water.orb_scenes(tl,t,hand,(240,160),17,None)
    sheets.apply_scenes(scenes,form,tl,t,hand,17)
    a,b=scenes[30],scenes[38]
    assert a.sheets and b.sheets
    assert np.mean(abs(a.sheets[0].xyz-b.sheets[0].xyz))>.5
    assert all(blob.free and blob.r<4 for blob in a.blobs)
    assert all(np.isfinite(sf.xyz).all() for sf in a.sheets)


def test_silent_and_unobserved_hands_cannot_launch_actions():
    t=track();xy=np.asarray(t['xy']);xy[20:31,0]+=np.linspace(0,80,11);xy[31:,0]+=80;t['xy']=xy.tolist()
    assert not sheets.choreography(t,timeline(holds=()),'ribbon')
    t['conf']=[.5]*72
    assert not sheets.choreography(t,timeline(hits=(25,)),'jet')


@pytest.mark.parametrize('axis,sign,action',[(0,1,'water_blade'),(1,-1,'rising_wave'),(1,1,'splash_crown')])
def test_each_measured_stroke_has_one_distinct_response(axis,sign,action):
    t=track();xy=np.asarray(t['xy']);xy[20:31,axis]+=sign*np.linspace(0,80,11);xy[31:,axis]+=sign*80;t['xy']=xy.tolist()
    result=sheets.choreography(t,timeline(),'ribbon')
    assert len(result)==1 and result[0]['action']==action
    assert result[0]['source']=='hand'


def test_repeated_voice_pulses_have_separate_births_and_stop():
    t=track();tl=timeline(hits=(15,40));hand=water.hand_of(t,72)
    actions=sheets.choreography(t,tl,'jet')
    assert [a['frame'] for a in actions]==[15,40]
    assert all(a['action']=='pulse_jet' for a in actions)
    assert not sheets.action_surfaces(actions[0],14,hand,tl,17)
    assert sheets.action_surfaces(actions[0],25,hand,tl,17)
    assert not sheets.action_surfaces(actions[0],45,hand,tl,17)


def test_dim_observed_hand_can_launch_voice_without_inventing_a_gesture():
    t=track();t['conf']=[.67]*72
    actions=sheets.choreography(t,timeline(hits=(40,)),'ribbon')
    assert len(actions)==1 and actions[0]['source']=='voice'
    t['src'][40]='lost'
    assert not sheets.choreography(t,timeline(hits=(40,)),'ribbon')


def test_released_water_does_not_follow_the_hand_and_obeys_camera_motion():
    t=track();tl=timeline(hits=(15,));hand=water.hand_of(t,72);action=sheets.choreography(t,tl,'jet')[0]
    first=sheets.action_surfaces(action,25,hand,tl,17)[0].xyz.copy()
    hand.p[20:]+=200
    assert np.allclose(first,sheets.action_surfaces(action,25,hand,tl,17)[0].xyz)
    tl.camera=np.repeat(np.eye(3)[None],72,0);tl.camera[25,0,2]=13
    shifted=sheets.action_surfaces(action,25,hand,tl,17)[0].xyz
    assert np.allclose(shifted-first,[13,0,0],atol=1e-4)


def test_launched_sheet_survives_loss_of_the_emitter(monkeypatch):
    t=track(72);t['scale']=[20.]*72;t['src'][16:]=['lost']*56;t['conf'][16:]=[0.]*56
    tl=timeline(72,holds=((4,14),),hits=(12,));plate=np.full((72,240,160,3),90,np.uint8)
    hand=water.hand_of(t,72);hand.presence[16:]=0
    monkeypatch.setattr(water,'hand_of',lambda *args:hand)
    out=water.render(plate,t,None,tl,'jet',17)
    assert np.any(out[30]!=plate[30])
    assert np.array_equal(out[65],plate[65])  # the jet and its finite spray have both drained


def test_surface_render_preserves_occluded_hand_and_pixels_outside_its_box():
    t=track(1);hand=water.hand_of(t,1);plate=np.full((1,240,160,3),80,np.uint8)
    xyz=np.array([[[40,50,-100],[40,100,-100]],[[110,50,-100],[110,100,-100]]],np.float32)
    surface=sheets.Surface(xyz);sc=water.Scene(sheets=[surface]);room=water.Room(plate);noise=water.Noise(4)
    mask=np.full((240,160),255,np.uint8)
    out=water.frame(0,plate[0],mask,sc,hand,room,noise,0,4)
    assert np.array_equal(out,plate[0])
    surface.xyz[:,:,2]=100
    a=water.frame(0,plate[0],None,sc,hand,room,noise,0,4)
    b=water.frame(0,plate[0],None,sc,hand,room,noise,0,4)
    assert np.array_equal(a,b) and not np.array_equal(a,plate[0])
    assert np.array_equal(a[120:],plate[0,120:]) and np.array_equal(a[:40],plate[0,:40])


def test_forms_are_selectable_with_foley_and_hand_emitters():
    from cuesync.bending import MADE, MADE_FOLEY, EMITTER_OF
    for form in FORMS['water']:
        effect='water_'+form
        assert MADE[effect]==('water',form)
        assert effect in MADE_FOLEY and EMITTER_OF[effect]=='palm'


def test_new_forms_drain_after_voice_and_have_no_infinite_spray():
    t=track();tl=timeline(holds=((4,25),));hand=water.hand_of(t,72)
    for form in sheets.NEW_FORMS:
        scenes=[water.Scene() for _ in range(72)]
        sheets.apply_scenes(scenes,form,tl,t,hand,17)
        assert scenes[15].sheets
        assert not scenes[31].sheets and not scenes[60].blobs
