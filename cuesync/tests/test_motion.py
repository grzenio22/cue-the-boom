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


def test_short_detector_gap_follows_visible_pixels_without_enabling_strokes():
    import cv2
    from cuesync.motion import bridge_short_gaps
    rng=np.random.default_rng(3);texture=rng.integers(40,230,(36,36,3),dtype=np.uint8)
    frames=np.zeros((18,100,140,3),np.uint8);mask=np.zeros(frames.shape[:3],np.uint8)
    for k in range(18):
        frames[k,40:76,30+k*2:66+k*2]=texture;mask[k,40:76,30+k*2:66+k*2]=255
    t=track(np.tile([48.,58.],(18,1)));t['src'][1:]=['lost']*17;t['conf'][1:]=[0.]*17
    out=bridge_short_gaps(t,frames,mask)
    assert out['flow_bridge_frames']==list(range(1,13))
    assert np.linalg.norm(np.asarray(out['xy'][12])-[72,58])<1
    assert not analyze(out).events
    assert t['src'][1]=='lost' and t['xy'][12]==[48.,58.]
    mask[3:]=0
    absent=bridge_short_gaps(t,frames,mask)
    assert absent['flow_bridge_frames']==[1,2]


def test_gap_recovery_measures_rotation_and_scale_without_altering_input():
    import cv2
    from cuesync.motion import bridge_short_gaps
    rng=np.random.default_rng(41)
    first=np.zeros((240,280,3),np.uint8)
    first[70:170,90:190]=rng.integers(30,230,(100,100,3),dtype=np.uint8)
    matte=np.zeros(first.shape[:2],np.uint8);matte[70:170,90:190]=255
    matrix=cv2.getRotationMatrix2D((140,120),9,.88);matrix[:,2]+=[12,-18]
    frames=np.stack([first,cv2.warpAffine(first,matrix,(280,240))])
    masks=np.stack([matte,cv2.warpAffine(matte,matrix,(280,240))])
    t=track(np.tile([140.,120.],(2,1)));t['scale']=[100.,100.]
    t['src'][1]='lost';t['conf'][1]=0
    recovered=bridge_short_gaps(t,frames,masks)
    assert recovered['flow_bridge_frames']==[1]
    assert np.linalg.norm(np.asarray(recovered['xy'][1])-[152,102])<2
    assert abs(recovered['scale'][1]-88)<3
    assert t['scale'][1]==100 and not analyze(recovered).events
