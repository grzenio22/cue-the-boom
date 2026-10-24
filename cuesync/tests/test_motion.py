"""Only sustained, observed motion relative to the room can trigger choreography."""
import numpy as np
from cuesync.motion import analyze


def track(xy):
    n=len(xy)
    return dict(xy=np.asarray(xy).tolist(), scale=[40.]*n, conf=[1.]*n,
                src=['mp']*n, gesture=['Open_Palm']*n, fps=24)


def swipe():
    xy=np.tile([80.,130.],(70,1))
    xy[20:31,0]+=np.linspace(0,65,11)
    xy[31:,0]+=65
    return xy


def test_directional_strokes_and_one_event_per_stroke():
    for axis,sign,kind in [(0,1,'swipe'),(1,-1,'lift'),(1,1,'slam')]:
        xy=swipe(); displacement=xy[:,0]-80
        xy[:]=[80,130];xy[:,axis]+=sign*displacement
        events=analyze(track(xy)).events
        assert len(events)==1
        assert events[0].kind==kind and 20<=events[0].frame<=30
        assert sign*events[0].direction[axis]>.95


def test_camera_translation_and_zoom_cannot_trigger_a_projectile():
    xy=swipe();cam=np.repeat(np.eye(3)[None],len(xy),0)
    cam[:,0,2]=xy[:,0]-80
    assert not analyze(track(xy),cam).events
    zoom=np.linspace(1,2,len(xy));cam[:,0,2]=0
    cam[:,:2,:2]*=zoom[:,None,None]
    t=track(np.array([80.,130.])*zoom[:,None]);t['scale']=(40*zoom).tolist()
    assert not analyze(t,cam).events


def test_tracking_spike_and_reacquisition_are_not_swipes():
    xy=swipe();xy[:,0]=80;xy[24,0]=200
    assert not analyze(track(xy)).events
    t=track(swipe());t['src'][19:33]=['lost']*14
    assert not analyze(t).events
    t=track(swipe());t['conf'][19:33]=[.5]*14
    assert not analyze(t).events


def test_second_distinct_stroke_rearms_and_has_plain_receipt():
    xy=swipe();xy[48:59,0]-=np.linspace(0,65,11);xy[59:,0]-=65
    result=analyze(track(xy))
    assert len(result.events)==2
    assert result.events[1].direction[0]<-.95
    import json
    assert 'swipe' in json.dumps(result.receipt())
