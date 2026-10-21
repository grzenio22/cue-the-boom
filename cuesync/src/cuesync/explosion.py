"""An explosion thrown from a strike (cuesync.elements' contract). A fireball rendered ahead by LTX-2.5 + the
ainvfx-fluid IC-LoRA over black (a white-hot flash bursting into a rolling ball of flame that turns into thick black
smoke, rising and drifting) is set off ahead of the fist on the strike and from then on is fixed in the room, never
carried on the hand: the fist pulls back from it, the hand can open beneath its smoke. Everything that touches the
plate is made here on the CPU:

- the white-hot core on the strike frame, its glow, and the lens's warm veil over the whole picture
- the exposure flash (+0.45 stop near the blast) and the camera's knock and zoom punch
- the shock front: a ring racing out that bends the room behind it (red and blue a little apart) with lit dust on it
- sparks flung off the ball (blackbody-cooling comets with drag and gravity, some flying past the arm) and embers
  floating up in the smoke
- the fireball's light on the room, from its own image (its fire spread round it, tinted orange, flickering, handed
  over from the flash's and dying as it turns to soot; the wall by the burning ball never as bright as the ball), and
  on the arm a warm rim on the outline facing it, a warm-white edge while the ball burns, the skin keeping its hue,
  its face hardly lit (the ball is behind it)
- the ball rolling upward as it burns, its smoke climbing on out of the frame, thinning from its crown and drawing in
  from its underside
- the fire drawn as a light: exposed as a phone exposes for the room (white-yellow at its hottest, red-orange folds),
  its rendered billows kept inside translucent, turbulent margins with the phone's softness and grain, its
  highlights blooming round it; dying, still one body of fire over its own brown-black soot, its embers glowing in the
  soot round them
- the smoke's outer lobes eaten away into crisp wisps that rise and curl with it
- the smoke's billows sharpened and lit from above, its soot filled in over frames as the fire dies (not in one), the
  render's own billows on it as it takes over from the fire, scattering the room's light as well as hiding what is
  behind it, in that light's colour
- the hot air shimmering round the ball

The rendered fireball and its smoke are composited by cuesync.composite (smoke as dense soot relit by the room), its
dark folds read as soot rather than holes, its fire then drawn over it on a camera's fire colours for its brightness,
and the arm's matte in front of it all.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import composite as C
from . import fx_light as fx
from .transfer import lin

FORMS = ("fist", "palm")  # a punch throws it ahead of the fist along the forearm; a thrust palm blooms it past the palm

SIZE = 2.0  # the fireball's radius at its fullest, hand sizes (a "pf" plays it at HIT_SIZE's smallest)...
MOST = 0.40  # ...and at most this share of the frame's short side
CLOUD_MOST = 0.75  # its smoke at its widest (swollen) reaches at most this share of the short side from the blast
# (a render's cloud grows to twice its ball and more: sized by its ball alone it buried the whole picture)
THROW = (0.8, 2.5, 0.12)  # hands ahead of the fist it goes off: the strike's peak speed (px a second) over the hand times
# the third, between the first two (a harder punch throws it further); without a measured strike, 1 hand
PALM_THROW = 0.6  # a thrust palm's blast goes off this far past the palm's middle
EDGE = 0.10  # it goes off at least this share of the frame inside its edges
EDGE_FADE = 0.08  # share of the render's short side over which its own frame edges fade out
BLAST_SHARE = 0.01  # the render's blast is its first frame lit over this share of its most lit frame's area
GROW_FRAMES, BURST_SPEED = 8, 2.0  # the render's first frames of growth play this much faster (a blast is quick)
SWELL = (0.15, 10.0)  # after its burst the ball swells by this share over about SWELL[1] frames
ROLL = (0.15, 6.0, 0.3, 12.0)  # and rolls upward: its middle rises this many of its radii, eased in over about this
# many frames, and it and its smoke climb on at this many radii a second, getting going over about this many frames,
# out of the top of the frame (the render's ball hung where it went off, a flat disc fading in place; a fireball is
# buoyant)
TOP_STRETCH = 0.2  # its top half grows this much taller than its bottom over SWELL[1] (a rising ball's crown rolls
# up and out, its underside hardly moves)
CLEAR = (40, 45)  # frames after the blast its smoke starts to clear, and how long it takes, thinnest first (a
# room-filling soot that never cleared held a flat dark sheet over the room to the end of the shot)

CORE = {0: 1.0, 1: 0.5, 2: 0.15}  # the white-hot core's strength on the strike frame and after
CORE_LIGHT = {0: 1.0, 1: 0.6, 2: 0.4, 3: 0.25, 4: 0.12, 5: 0.05}  # its light on the room, outlasting it while the
# ball grows
FLASH = {0: 0.45, 1: 0.25, 2: 0.1, 3: 0.03}  # exposure, stops: the flash (at +0.7 the whole picture washed out; an
# iris overshooting after it dipped the room's light for a frame)
LIGHT_HOLD = 0.75  # retain this share of the previous firelight between frames to avoid a second flash during
# handoff. LIGHT_SETTLE releases the superseded broad flash first; subsequent firelight keeps this decay.
FLICKER = (0.12, 0.5)  # the fireball's light flickers by about this share, its noise smoothed over this many frames
# (8-12 Hz: a smooth ramp down is not how firelight behaves), as long as it is fire
FLASH_REACH, FLASH_FAR = 2.5, 0.35  # the flash falls off over this many fireball radii, to this share far from it
GROW_BLUR = {0: 0.3, 1: 0.22, 2: 0.12, 3: 0.07, 4: 0.04}  # the fireball's growth over the shutter, per frame
LIGHT_GAIN = 6.6  # the fireball lights the room from its own image: its fire's luminance spread round it
# (blast_light, LIGHT_FALLOFF), times this, on each surface's colour (fx.albedo): the wall within a radius of a ball at
# full burn lifts about +50 levels in red, green under half of that, blue not at all, and the light dies with the fire
# (the room light of a point source a hand behind the ball lifted the wall +9 levels: the blast lit nothing)
LIGHT_FALLOFF = (0.8, 4.2)  # source-convolved spread: (1 + r / (this many ball radii)) ** -power. Sustained fire
# needs a compact pool: the camera shoulder otherwise flattens both the nearby and distant wall to the same orange.
LIGHT_SETTLE = (4, 8, 2.0)  # frames after ignition: keep the initial flash's broader falloff (last value), then
# tighten it continuously and restore the skin's recorded shadow relief as the camera settles onto the live fire.
LIGHT_V = (0.5, 0.7)  # its colour is a camera's fire colour (RAMP) for the fire's brightness, held between deep
# orange and orange (the render's own lemon-yellow lifted a khaki wall green; yellower, the wall went yellow-white)
LIGHT_GLOW = 0.03  # the same light scattered in the air and the lens, over everything
LIGHT_KNEE = 0.9  # linear: the lit room rolls off from here to white (a channel lit past it clipped to a flat plateau)
LIGHT_WARM = 0.45  # on a bright wall red soon reaches white and stops rising: the light's green then rises at most this
# much of its red, and its blue not at all (at 0.8 a khaki wall by the ball went peach, not orange; at 0.5, with blue
# rising half of green, the wall by the ball still read a little yellow-pink)
ARM_LIGHT = (0.05, 1.8, 3.0)  # on the arm: a little on its face (the blast is behind it: the fist that hides the ball
# shows the camera its unlit front), this much in a rim on its outline facing the fire (rim_light), and this much more
# while the fire burns (BURN)
RIM_BAND = (1.0, 12.0)  # px at 1080: the rim starts this far inside the arm's outline (the contact edge catches firelight)
# and dies away over this many more (a tenth of it left 30 px in)
RIM_FACE = (0.3, 0.8)  # the outline lit is the one facing the fire: its outward normal on the way to the fire over this
SKIN_GR = 0.75  # on skin the light keeps the skin's hue: green rises to at most this of red...
SKIN_BG = 0.15  # ...and blue at most this much of green's rise (lit skin's red is near white: green rising alone
# turned a fist lemon-yellow, a rubber glove)
SKIN_LIFT = (10.0, 40.0, 30.0)  # display levels: the most the fire lifts the arm's face, and its rim, and the rim this
# much more while the fire burns (lit flat, a fist in front of the ball read cream-white; held to +40, the fist in
# front of an orange fireball read unlit)
SKIN_SHADE = 0.65  # display luminance of a fully lit skin face; darker native folds receive proportionally less
# added rim light, and less of the warm-white rim, so a common highlight target cannot fill the hand's shadows.
SKIN_RIM = (0.88, 0.45)  # ...where on its rim green may rise to this share of red and blue to this share of green's
# rise (skin's red is near white already: held to SKIN_GR its rim could rise a dozen levels, and a rim lit by a
# white-hot ball goes warm white)
HOT_RIM = (0.65, (0.25, 0.65), (255, 225, 180))  # while the ball burns, the arm's rim (where rim_light is over the first
# to the second of these, a band about 4-14 px in) goes up to this much of the way to this warm white (display RGB):
# skin's red is already clipped, so light added there rose in green alone and read peach, not a hot edge
KNOCK_PX, ZOOM = 14.0, 0.022  # the camera's knock (px at 1080) and its push from the pressure wave (share)
DEFOCUS_PX = 1.2  # px at 1080: the blast is further away than the fist the phone focused on
SOOT_ALBEDO = 0.25  # its smoke is black: it takes this share of the room's light...
SOOT_LIT = 0.3  # ...the fireball's light among it, at this share of what lights the wall (lit by the room alone, the
# soot hugging the ball was black, a drop shadow; lit by it, it glows brown-orange round the fire, then dies with it)
SOOT_FILL = (0.16, 0.12)  # display levels (black taken off): the cloud's dark body is raised to the first inside its
# silhouette, its folds kept round it (raised to one flat level, the inside of the cloud lost every billow), and none
# of it left under the second (the composite reads that as dense: a fold under it let the wall through)
BILLOW = (0.5, 1.7, 0.6)  # the render's own light and shade on its billows, put back on the filled cloud: its
# smoke's luma over their median to this power, in [lo, hi] (a dim cloud filled to one grey read as a flat dark disc;
# clipped hard, its folds read as a camouflage print)
BILLOW_IN = 0.02  # share of the short side: kept this far inside the cloud's outline (its thin rim is the room's)
SOOT_CLOSE = 0.03  # share of the short side: gaps narrower than this inside the cloud are closed before it is filled
SOOT_RAMP = 1.0 / 8  # the filled silhouette changes by at most this a render frame (as the fire dies its body stops
# setting the cloud's threshold and gaps close: the soot snapped dense in one frame, +75% dark area)
DETAIL = (0.8, 6.0)  # the smoke's billows sharpened: its luma plus this much of what a blur of this many px at 1080
# takes off it (composited, the cloud read out of focus, a stain on the wall)
TOP_LIGHT = (0.5, 0.012)  # the room's light from above: a billow's upper faces lifted by up to this share (about
# 10-15 levels), its slopes read off the render's smoke blurred this share of the short side (lit evenly, the
# cloud's inside was flat camouflage blotches, not a volume)
# The fire drawn as a light (Fire): composited as the render draws it, the ball sat at luma 166 on a wall it lit to 190
# (a printed disc), a pale line and a dark hairline round it and a brown band of soot outside it (a sticker and its
# drop shadow), its tongues pale and see-through, and the embers in the soot ringed dark
BURN = (0.2, 0.6)  # it burns while the render's fire is over the first share of its brightest frame's, fully from the
# second: its ball is then one shape with nothing round it but the wall it lights (its soot comes as it dies)
BODY = (0.2, 0.5, 10.0, 0.4, 7.0)  # as it dies it is still one body of fire with one edge, over its own soot, while
# the render's fire covers over the first share of its widest frame's area, fully from the second, its edge feathered
# this many px at 1080 more as it goes into its smoke, its smoky folds up to this many of its radii across part of it,
# its strands thinner than about twice this many px at 1080 not (its dim fire drawn over the composite's relit smoke,
# the dying ball read as melted wax: raspberry folds in contour bands, a crisp orange stroke round it)
HANDOFF = 0.125  # after their peak the weights change by at most this per render frame. Fire anticipates its last
# flame; the dense body continues into the first smoke frames on the smoke's moving silhouette, cooling gradually.
MATERIAL_RELEASE = 0.08  # the moving fire key releases over about thirteen frames: a saturation threshold must not
# replace dark fire folds with light grey soot at once. Source RGB stays current and sharp throughout the handoff.
BODY_SOOT = (0.05, (255, 107, 36), 0.6)  # inside it, its soot: this share of the lit room's luminance in this colour
# (display RGB, brown-black: its darkest folds near (80, 35, 12) on a lit wall), its smoke's own brightness taken down
# the fire's colours to this share (red-brown folds that keep the render's texture)
EXPOSE = (2.3, 1.6, 0.3)  # the fire's brightness as a phone exposed for the room shows it: w = 1 - (1 - v) ** the
# first, then w ** (1 + the second (1 - w)), on RAMP (near white-yellow at its hottest, its folds kept a deeper orange),
# and while it burns lifted towards white by the third share of what it lacks (its darkest folds red-orange: unlifted
# they went black, a charred ball)
BALL = (0.5, 2.5, 2.0)  # the burning ball: where the render is fire (weight over the first), holes filled, shrunk this
# many px at 1080 and feathered over this many (its colour and its alpha end on one contour)
FLAME_EDGE = (100.0, 140.0, 2.4)  # px at 1080: irregular translucent flame paths across at most this depth, displaced by
# this much turbulent motion, with this lens softness. Only the margin changes; a filled hot contour read as a cutout.
FIRE_RESOLVE = (3.0, 18.0, 0.22)  # px at 1080: suppress unresolved fine texture, then mix this share of broader
# flame brightness into it. The phone keeps the large billows but cannot resolve the source's crisp, dark filigree.
LIMB = (8.0, 4.0, 14.0, 0.8, 0.08)  # its limb's folds filled up to the fire round them (blurred this many px at 1080),
# fully to this many px at 1080 in from its edge, none past the second, its darkest folds there lifted to about this
# (RAMP: yellow-orange, as bright as the wall it lights), smoothly over this (a light's limb is never darker than the
# wall it lights: unlifted, a dark rim drew the outline; all of it lifted to one yellow level, an even band drew it, an
# outer glow, its rays bright, dark, then bright)
WALL_CAP = (0.3, 0.9, 0.35, 0.6)  # the wall within this many of its radii past its edge is lit to at most this share
# of its limb's luma (a light dimmer than the wall it lights reads as a printed disc), the light at its edge kept to at
# least this share of itself and held less further out, none from this many radii out (held all over, the room the
# ball lit went dim with it)
BLOOM = (0.82, ((8.0, 0.5), (16.0, 0.35)))  # its highlights (linear luminance over the first) spread by the lens over
# these blurs (px at 1080) at these gains: wide beside white-hot fire, none beside its folds (a glow of all of its
# light round the ball read as a layer style's even outer glow)
EMBER_GLOW = (((8.0, 0.4), (24.0, 0.6)), 1.5, (1.0, 0.45, 0.1), 0.3)  # as it dies, its fire glows into the soot
# round it: its luminance spread over these blurs (px at 1080, weights: about an exponential falloff over 25 px), times
# this, in this light's colour (linear), added (mixed towards the soot's colour, each ember's glow went yellow, orange,
# mauve); the dying ball's own fire, and over it, this share as much (all of it, a wide orange halo stood round the
# ball and its folds went orange)
SOOT_BANDS = (6.0, 24.0, 9.0, (28, 36))  # as the fire turns to soot its billows come from the render's own: a
# band-pass of its smoke's log luma between blurs of this many and this many px at 1080, its spread this many display
# levels (std), from the fire's burning out to this many frames after the blast, fading by the next (the soot was one
# grey, a felt, until the later smoke's own billows took over)
# The smoke's outline (Outline), in radii of the cloud: drawn as rendered, its edge was a broad, soft, round dark line
# (it advertised the composite)
OUTER = (0.42, 0.92, 0.7)  # the smoke's outer lobes (from its edge to this deep, in its radii) are eaten away, up to
# this share of it at its edge, the share rising to it as its depth to this power (feathered instead, the cloud read out
# of focus, a stain on the wall)...
WISPS = (0.035, 0.05, 0.03, 0.3)  # ...between its billows (the render's own light) and wisps of a noise this big across
# and up (shares of the short side) fixed in the cloud and rising with it, cut this sharply (a share of their range:
# edges 2-4 px across at 1080), the billows this share of what decides
LOBE = (0.02, 0.4, 20, 40)  # share of the short side: what is left thickens from its cut edge over this far, as a
# round lobe's path through it does (cut flat, it read as paper); young soot over this share of it, growing to all of
# it from this many frames after the blast to this many (as the fire turned to soot its edge faded over 20-25 px: felt)
WISP_CURL = (0.04, 0.06, 40)  # the wisps curl: swirled up to this share of the short side, in eddies this big, over
# this many frames
TOP_THIN = (14, 50, 0.6)  # from this many frames after the blast to this many, the smoke thins from its crown, by this
# share at the top (it rises and thins from the top, not fading in place)...
BOTTOM_THIN = (6, 24, 0.75)  # ...and from its underside about twice as fast, eaten in from its edge (its lobes hung
# below it like ink drips)
OUTER_SOFT = 2.5  # px at 1080: what the smoke does to the room is softened this much in its outer lobes (the render
# draws a thin bright rim round its smoke: composited it drew a contour line round the cloud)
SCATTER = (1.5, 0.12, 6.0, 0.15, 3e-3)  # the soot scatters the room's light as well as hiding what is behind it:
# where the composite leaves a share f of the room, of what it lets through from behind (fitted over windows the second
# share of the short side) only f ** the first still comes through, blurred by up to the third px at 1080 as it
# thickens, the rest the room's light round it (blurred the fourth share of the short side) scattered back (as the
# composite drew it, a white switch plate kept its edges and a 54 level lead over the smoked wall beside it: in front
# of the cloud); the fit takes only what varies more than the fifth (a variance, linear luminance) as what is behind
SOOT_HUE = 0.85  # black soot takes the colour of the light it scatters and of what shows through it: the room's (lit by
# the fire), this much of its colour kept, the rest grey (relit from the render's dim red fire, the soot by the dying
# ball went raspberry and mauve, and neutral grey over the warm wall read lavender)
VOLUME_HANDOFF = (16, 26)  # frames after ignition: open the cooling soot's unequal optical paths while its last hot
# pockets remain. Delaying until 32 left the dense phase's thin lobes sealed by the older opaque composite.
VOLUME_DEPTH = 3.4  # optical depth through overlapping soot: background detail fades before its broad illumination,
# so a bright switch cannot remain a pale rectangular insert inside a thin lobe.
LOBE_SEPARATE = (28, 36)  # after the cooling handoff, release the filled interior into overlapping lobes. Each
# lobe keeps its own dense core and thin edge; the low-density wisps disperse faster instead of fading the body flat.
PARAMS = dataclasses.replace(  # composite.composite_frame's for a fireball rolling into black smoke (its light is ours)
    C.CompositeParams(), light_gain=0.0, fire_gain=0.75, shading_lo=0.5, shading_hi=1.35, interior_blur_frac=0.04,
    interior_lo=0.45, interior_hi=0.9, sat_lo=0.4, sat_hi=0.65)

# fire as a phone films it, by its brightness (display values): deep red at its dim edges, an orange body, yellow,
# then white-hot (a luminance-true blackbody read tan: a phone's tone curve crushes the blue a flame has)
RAMP_V = np.array([0.0, 0.15, 0.35, 0.55, 0.72, 0.86, 0.95, 1.0])
RAMP = np.array([[0, 0, 0], [92, 20, 5], [196, 62, 10], [242, 118, 18], [255, 165, 40], [255, 208, 92],
                 [255, 238, 170], [255, 250, 225]], np.float64) / 255.0

SHOCK_FRAMES = 12  # the front crosses the frame in half a second (shorter read as a cut, not a wave)
SHOCK_PX = 20.0  # px at 1080 of refraction at its strongest
SHOCK_WIDTH = 0.035  # share of the short side: the front's thickness at the start (it thickens as it travels)
SHOCK_DUST = 0.2  # opacity of the lit dust riding the front (it reads on a flat wall, where there is nothing to bend)
SHOCK_CA = 0.35  # red and blue bent this much more and less than green
SPARKS, EMBERS = 55, 45  # sparse hot debris; a dense simultaneous spray drew radial speed lines at ignition


# ------------------------------------------------------------------ where it goes off, and the render's own blast
def blast_point(track: dict, k: int, form: str = "fist", aim: float | None = None
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """(point px, aim, hand size) on the strike frame ``k``. A fist: its leading landmark pushed THROW hands further
    along the forearm (wrist to middle knuckle); a palm: its middle pushed PALM_THROW. Without landmarks, the track's
    emitter and direction. The studio's arrow overrides the hand's direction."""
    s = float(track["scale"][k])
    lm = (track.get("landmarks") or [None] * (k + 1))[k]
    arrow = None
    if aim is not None:
        a = math.radians(aim)
        arrow = np.array([math.sin(a), -math.cos(a)])
    if lm is None:
        p = np.asarray(track["xy"][k], np.float64)
        aim = np.asarray(track["dir"][k], np.float64) if arrow is None else arrow
        return p + (PALM_THROW if form == "palm" else 1.0) * s * aim, aim / max(np.hypot(*aim), 1e-6), s
    lm = np.asarray(lm, np.float64)[:, :2]
    aim = lm[9] - lm[0] if arrow is None else arrow
    aim /= max(float(np.hypot(*aim)), 1e-6)
    if form == "palm":
        return lm[[0, 5, 9, 13, 17]].mean(0) + PALM_THROW * s * aim, aim, s
    speed = max((float(e.get("peak_speed") or 0.0) for e in track.get("events", []) if abs(int(e["frame"]) - k) <= 4),
                default=0.0)
    throw = float(np.clip(THROW[2] * speed / max(s, 1.0), THROW[0], THROW[1])) if speed > 0 else 1.0
    return lm[int(np.argmax(lm @ aim))] + throw * s * aim, aim, s


def inside(p: np.ndarray, w: int, h: int) -> np.ndarray:
    """``p`` moved (as little as it takes) to EDGE of the frame inside its edges."""
    return np.array([np.clip(p[0], EDGE * w, (1 - EDGE) * w), np.clip(p[1], EDGE * h, (1 - EDGE) * h)])


def render_blast(element: np.ndarray, floor: int) -> tuple[int, np.ndarray, float]:
    """(its first lit frame, where the blast is on it, the ball's radius GROW_FRAMES later) in a render over black:
    measured, so any render of a fireball can play (its blast is not always where its guide put it)."""
    lit = C._peak(element[:, ::4, ::4]).astype(np.int16) > floor + 24
    counts = lit.reshape(len(element), -1).sum(1)
    # (a spark held where the blast will be before it goes off is not the blast: a render can hold one for frames)
    on = np.nonzero(counts >= max(4, BLAST_SHARE * counts.max()))[0]
    if on.size == 0:
        raise ValueError("the explosion's render shows no blast")
    k = int(on[0])
    ys, xs = np.nonzero(lit[k])
    centre = np.array([4.0 * xs.mean(), 4.0 * ys.mean()])
    r8 = 4.0 * math.sqrt(float(counts[min(k + GROW_FRAMES, len(element) - 1)]) / math.pi)
    return k, centre, max(r8, 8.0)


def render_reach(element: np.ndarray, floor: int, centre: np.ndarray, start: int) -> float:
    """How far from ``centre`` the render's fire and smoke reach at their widest from frame ``start`` on (px; the
    98th percentile of its lit pixels' distances, so a stray wisp does not count)."""
    h, w = element.shape[1:3]
    yy, xx = np.mgrid[:h:4, :w:4]
    d = np.hypot(xx - centre[0], yy - centre[1])
    reach = 0.0
    for t in range(start, len(element), 4):
        lit = C._peak(element[t, ::4, ::4]).astype(np.int16) > floor + 6
        if lit.sum() > 20:
            reach = max(reach, float(np.percentile(d[lit], 98)))
    return reach


def fire_burns(element: np.ndarray, floor: int) -> tuple[np.ndarray, np.ndarray]:
    """Per render frame, in [0, 1]: how far its fire burns, its fire's light (at an eighth of its size) over that of its
    brightest frame between BURN's shares; and how far its fire is still one body, its area over that of its widest
    frame between BODY's."""
    light, area = np.zeros(len(element)), np.zeros(len(element))
    for k, f in enumerate(element[:, ::8, ::8]):
        e = C._black_floor_f(f, floor)
        wf = C.fire_weight_map(e, PARAMS, f.shape[1], True)
        light[k] = float((C._lin(e) @ C._LUMA * wf).sum())
        area[k] = float((wf > 0.5).sum())
    weights = [C._smoothstep(BURN[0], BURN[1], light / max(float(light.max()), 1e-9)),
               C._smoothstep(BODY[0], BODY[1], area / max(float(area.max()), 1e-9))]
    for index, weight in enumerate(weights):
        peak = int(np.argmax(weight))
        if index == 0:
            for k in range(len(weight) - 2, peak - 1, -1):
                weight[k] = min(weight[k], weight[k + 1] + HANDOFF)
        for k in range(peak + 1, len(weight)):
            weight[k] = np.clip(weight[k], weight[k - 1] - HANDOFF, weight[k - 1] + HANDOFF)
    return tuple(weights)


def soot_shading_ref(element: np.ndarray, floor: int, start: int) -> float:
    """The median linear luma of the render's smoke (not its fire) from frame ``start`` on: its billows are lit and
    shadowed round this. (composite.smoke_shading_ref measures only smoke the render draws bright, and a black cloud
    is dim all through: its every billow sat at the bottom of the shading's range, a flat dark disc.)"""
    vals = []
    for f in element[start::max(1, (len(element) - start) // 12), ::4, ::4]:
        e = C._black_floor_f(f, floor)
        y = e @ C._LUMA
        m = (y > 0.04) & (C.fire_weight_map(e, PARAMS, f.shape[1], True) < 0.1)
        if m.sum() > 50:
            vals.append(C._lin(e[m]) @ C._LUMA)
    return float(np.median(np.concatenate(vals))) if vals else 0.25


def room_point(cam, k0: int, t: int, p) -> np.ndarray:
    """A point fixed in the room as seen on frame ``k0``, where it is on frame ``t`` (cam: bending.camera_path)."""
    from .bending import camera_between

    q = camera_between(cam, k0, t) @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


# ------------------------------------------------------------------ the shock front
def shock_ring(frame: np.ndarray, origin, aim, k: int, reach: float, side: int, seed: int,
               warm: np.ndarray, dust: np.ndarray) -> np.ndarray:
    """``frame`` (linear) seen through the blast's shock front ``k`` frames after it: a ring (a little stronger along
    the aim) that bends what is behind it outward ahead of the front and back behind it, red and blue a little apart,
    with lit dust riding it. ``warm``: the fireball's colour, which lights the ring early on."""
    h, w = frame.shape[:2]
    u = k / SHOCK_FRAMES
    R = 0.035 * side + reach * (1 - (1 - u) ** 1.6)
    strength = SHOCK_PX * side / 1080.0 * min(1.0, (k + 1) / 1.5) * (1 - u)
    width = SHOCK_WIDTH * side * (1 + 1.5 * u)
    ox, oy = float(origin[0]), float(origin[1])
    rng = np.random.default_rng([seed, 31])
    lobes = np.arange(3, 9)
    ph, amp = rng.uniform(0, 2 * np.pi, lobes.size), rng.uniform(0.5, 1.0, lobes.size) / lobes
    pad = 4 * width + strength + 4  # only the band round the front is touched
    y0, y1 = max(0, int(oy - R - pad)), min(h, int(oy + R + pad) + 1)
    x0, x1 = max(0, int(ox - R - pad)), min(w, int(ox + R + pad) + 1)
    if y1 <= y0 or x1 <= x0:
        return frame
    xs = np.arange(x0, x1, dtype=np.float32) - ox
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] - oy
    dist = np.sqrt(xs * xs + ys * ys) + 1e-3
    nx, ny = xs / dist, ys / dist
    ang = np.arctan2(ys, xs)
    wob = 1 + 0.04 * (amp[:, None, None] * np.sin(lobes[:, None, None] * ang + ph[:, None, None])).sum(0)
    cone = 0.6 + 0.4 * np.clip(nx * aim[0] + ny * aim[1], 0, 1)
    z = (dist - R * wob) / width
    bend = (strength * cone * z * np.exp(0.5 * (1 - z * z))).astype(np.float32)
    out = frame.copy()
    box = (y0, y1, x0, x1)
    for c, more in enumerate((1 + SHOCK_CA, 1.0, 1 - SHOCK_CA)):
        shift = np.stack([-nx * bend * more, -ny * bend * more], -1).astype(np.float32)
        out[y0:y1, x0:x1, c] = fx._remap_box(np.ascontiguousarray(frame[..., c]), shift, box)
    edge = SHOCK_DUST * (1 - u) ** 1.3 * cone * np.exp(-0.5 * ((z - 0.8) / 0.7) ** 2) * dust[y0:y1, x0:x1]
    room = fx._lit_tone(frame)
    hot = min(1.0, 2.5 * (1 - u) ** 3)
    tone = np.maximum(room * (1 - hot) + warm * hot * 0.6, room)  # never darker than the lit room: air catching light
    out[y0:y1, x0:x1] += edge[..., None] * (tone - out[y0:y1, x0:x1])
    return out


# ------------------------------------------------------------------ sparks and embers
class Particles:
    """What the blast throws, in the room as it was on the strike frame (carried by the camera after): sparks flung off
    the fireball's surface as it grows (drag, gravity, cooling along the blackbody line, drawn as comets over the
    shutter: a bright head, a fading tail) and embers that float up on the hot gas, flicker and fall."""

    def __init__(self, origin, aim, s: float, R: float, side: int, seed: int, count: float = 1.0):
        rng = np.random.default_rng([seed, 77])
        ns, ne = int(round(SPARKS * count)), int(round(EMBERS * count))
        n = ns + ne
        self.ember = np.arange(n) >= ns
        a_aim = math.atan2(aim[1], aim[0])
        fwd = rng.random(n) < 0.55
        ang = np.where(fwd, a_aim + rng.normal(0, 0.8, n), rng.uniform(-np.pi, np.pi, n))
        dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
        self.t0 = np.where(self.ember, rng.uniform(2, 12, n), rng.choice([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0], n))
        r_then = R * np.clip(0.3 + 0.7 * self.t0 / 4.0, 0, 1)  # thrown off the ball's surface as it grows
        r0 = np.where(self.ember, R * np.sqrt(rng.uniform(0, 0.7, n)), r_then * rng.uniform(0.55, 0.95, n))
        self.p0 = np.asarray(origin, np.float64)[None] + dirs * r0[:, None]
        streak = (rng.random(n) < 0.08) & ~self.ember  # a few fast streakers
        speed = s * np.exp(rng.normal(math.log(0.5), 0.5, n))
        speed = np.where(streak, s * rng.uniform(1.4, 2.2, n), speed)
        speed = np.where(self.ember, s * rng.uniform(0.03, 0.12, n), speed)
        self.v0 = dirs * speed[:, None]
        self.v0[self.ember, 1] -= s * rng.uniform(0.08, 0.2, int(self.ember.sum()))  # carried up by the hot gas
        self.tau = np.where(self.ember, rng.uniform(12, 25, n), rng.uniform(4, 10, n))  # drag time constant, frames
        self.g = s * np.where(self.ember, rng.uniform(0.002, 0.006, n), rng.uniform(0.02, 0.04, n))
        self.life = np.where(self.ember, rng.uniform(30, 75, n),
                             np.where(streak, rng.uniform(10, 18, n), rng.uniform(5, 20, n)))
        self.T0 = np.where(self.ember, rng.uniform(1700, 2100, n), rng.uniform(2400, 3300, n))
        self.tauT = np.where(self.ember, 200.0, rng.uniform(4, 10, n))
        self.I0 = np.where(self.ember, rng.uniform(0.5, 1.2, n),
                           np.where(streak, rng.uniform(4, 7, n), rng.uniform(1.2, 3.5, n)))
        self.size = np.where(self.ember, rng.uniform(1.6, 2.8, n), np.where(streak, 2.0, rng.uniform(1.0, 1.8, n)))
        self.size *= side / 1080
        self.front = rng.random(n) < 0.3  # flying towards the camera: in front of the arm and the fireball
        self.phase = rng.uniform(0, 2 * np.pi, n)

    def at(self, i: int, ages: np.ndarray) -> np.ndarray:
        """Particle ``i``'s positions at ``ages`` frames after the strike (room coordinates on the strike frame)."""
        tau, g = self.tau[i], self.g[i]
        decay = tau * (1 - np.exp(-ages / tau))
        fall = g * (tau * ages - tau * decay)
        return self.p0[i][None] + self.v0[i][None] * decay[:, None] + np.stack([np.zeros_like(fall), fall], -1)

    def draw(self, since: float, shape, cam_M: np.ndarray, front: bool, shutter: float = 0.5):
        """The light (linear) of the particles on one side of the arm, ``since`` frames after the strike, the room
        carried by ``cam_M`` (3x3: the strike frame to this one): (its light over the box round them, the box (y0, y1,
        x0, x1) on the (h, w) ``shape``), or None for none in the frame."""
        h, w = shape
        K = 8
        comets = []
        for i in range(len(self.ember)):
            if self.front[i] != front:
                continue
            age = since - self.t0[i]
            if age < 0 or age > self.life[i]:
                continue
            P = self.at(i, np.linspace(max(0.0, age - shutter), age, K))
            P = (cam_M @ np.c_[P, np.ones(len(P))].T).T
            P = P[:, :2] / P[:, 2:3]
            length = float(np.hypot(*(P[-1] - P[0])))
            # Phone exposure resolves a few short debris trails, not a fan of 100-270 px speed lines. Preserve
            # each particle's head and trajectory while limiting the light integrated along its fast tail.
            limit = 0.025 * min(h, w)
            if length > limit:
                P = P[-1] + (P - P[-1]) * (limit / length)
                length = limit
            fade = min(1.0, (self.life[i] - age) / 5.0) * min(1.0, (age + 1) / 1.5)
            T = 1100 + (self.T0[i] - 1100) * math.exp(-age / self.tauT[i])
            I = self.I0[i] * ((T - 900) / (self.T0[i] - 900)) ** 3 * fade
            if self.ember[i]:
                I *= 0.65 + 0.35 * math.sin(self.phase[i] + 1.7 * age) * math.sin(0.6 * self.phase[i] + 2.9 * age)
            I *= min(1.0, 5.0 / max(length, 1.0)) ** 0.35  # its light spread along the streak (streaks stay bright)
            comets.append((np.round(P).astype(int), fx.kelvin_rgb(T) * I, max(1, int(round(self.size[i])))))
        if not comets:
            return None
        pts = np.clip(np.concatenate([c[0] for c in comets]), -1, [w, h])
        pad = 48  # (what its glow reaches: only the box round them is drawn and blurred)
        x0, y0 = (max(0, int(v) - pad) for v in pts.min(0))
        x1, y1 = (min(lim, int(v) + pad + 1) for v, lim in zip(pts.max(0), (w, h)))
        if x1 <= x0 or y1 <= y0:
            return None
        emit = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        for P, col, th in comets:
            for j, (a, b) in enumerate(zip(P[:-1], P[1:])):  # a comet: the head (now) bright, the tail fading
                kk = ((j + 1) / (K - 1)) ** 1.5
                cv2.line(emit, (int(a[0]) - x0, int(a[1]) - y0), (int(b[0]) - x0, int(b[1]) - y0),
                         tuple(float(c) * kk for c in col), th, cv2.LINE_8)
        if emit.max() <= 0:
            return None
        emit = cv2.GaussianBlur(emit, (0, 0), 0.7)
        emit = emit + 0.45 * cv2.GaussianBlur(emit, (0, 0), 2.5) + 0.2 * cv2.GaussianBlur(emit, (0, 0), 10.0)
        return emit, (y0, y1, x0, x1)


# ------------------------------------------------------------------ the render, made ready
def soot_shape(frame8: np.ndarray, floor: int, close: int, soft: float) -> np.ndarray:
    """(h, w) in [0, 1]: a render frame's cloud as its soot is filled: its body (not the render's faint haze round it,
    judged against its smoke, not its fire: as the fire died the cloud's threshold fell and its gaps closed at once),
    gaps narrower than ``close`` px closed, every gap open to the outside left out, the rest filled, its edge feathered
    ``soft`` px."""
    y = C._peak(frame8).astype(np.float32) - floor
    smoke = (y > 4) & (C.fire_weight_map(C._black_floor_f(frame8, floor), PARAMS, frame8.shape[1], True) < 0.1)
    some = y[smoke] if smoke.sum() > 100 else y[y > 4]
    body = float(np.percentile(some, 90)) if some.size > 100 else 0.0
    if body <= 0:
        return np.zeros(y.shape, np.float32)
    lit = (y > max(10.0, 0.3 * body)).astype(np.uint8)
    k = max(3, close | 1)
    closed = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ff = np.pad(closed, 1)  # a ring of outside all round, so the fill reaches every gap open to the outside
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    sil = cv2.erode((ff[1:-1, 1:-1] != 2).astype(np.float32), np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(sil, (0, 0), max(soft, 0.5))


class SootShapes:
    """The render's cloud as its soot is filled (soot_shape, at a quarter of the render's size: it only says where the
    soot is filled, its edges are the composite's), render frame by render frame from its blast, each frame's moving at
    most SOOT_RAMP towards what that frame measures: the fill grows in and lets go over frames. Asked for in order (a
    frame further on is reached through those before it)."""

    def __init__(self, el: np.ndarray, floor: int, start: int, scale: float, side: int):
        self.el, self.floor, self.e, self.sil = el, floor, start - 1, None
        self.close = int(SOOT_CLOSE * side / scale / 4)  # (sizes in the plate's px, at a quarter of the render's)
        self.soft = 0.006 * side / scale / 4

    def at(self, e: int) -> np.ndarray:
        while self.e < e:
            self.e += 1
            f = self.el[self.e]
            raw = soot_shape(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4), interpolation=cv2.INTER_AREA),
                             self.floor, self.close, self.soft)
            self.sil = raw if self.sil is None else self.sil + np.clip(raw - self.sil, -SOOT_RAMP, SOOT_RAMP)
        return self.sil


def wide_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """``a`` (float32) blurred over ``sigma`` px, a wide blur at a size it loses nothing at (a third of it a px)."""
    k = int(sigma // 3)
    if k < 2:
        return cv2.GaussianBlur(a, (0, 0), sigma)
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / k), (w, h), interpolation=cv2.INTER_LINEAR)


def smoke_blur(a: np.ndarray, smoke: np.ndarray, sigma: float) -> np.ndarray:
    """``a`` blurred over ``sigma`` px from its smoke alone (``smoke`` in [0, 1] weighting each pixel): the fire's light
    kept out of the smoke round it (blurred with it, the soot by an ember read pale, a halo)."""
    return wide_blur(a * smoke, sigma) / np.maximum(wide_blur(smoke, sigma), 0.05)


def fill_soot(E8: np.ndarray, floor: int, box, sil: np.ndarray, smoke: np.ndarray) -> np.ndarray:
    """The render with its cloud's dark body read as dense soot, not holes: over black the self-shadowed underside of a
    black cloud renders black, and composited it let the wall through as pale patches. Inside the cloud's filled
    silhouette ``sil`` (over ``box``; ``smoke``: where it is smoke, not fire) the render is raised by what its smoke
    round it lacks of SOOT_FILL's first level, so its folds and billows keep their light and shade round it, and to at
    least its second; by the fire, to what the fire lights round it."""
    y0, y1, x0, x1 = box
    part = E8[y0:y1, x0:x1].astype(np.float32)
    sigma = 0.02 * min(E8.shape[:2])
    level, least = (floor + v * (255 - floor) for v in SOOT_FILL)
    raised = part + np.maximum(level - smoke_blur(C._peak(part), smoke, sigma), 0.0)[..., None]
    lift = np.maximum(np.maximum(raised, 0.85 * wide_blur(part, sigma)), least)
    out = E8.copy()
    out[y0:y1, x0:x1] = np.clip(part + (np.maximum(lift, part) - part) * sil[..., None] + 0.5, 0, 255).astype(np.uint8)
    return out


def soot_detail(E8: np.ndarray, floor: int, box, smoke: np.ndarray) -> np.ndarray:
    """In place over ``box``: the render's smoke (``smoke`` in [0, 1], not its fire) with its billows sharpened,
    DETAIL: its luma plus a share of what a blur of its smoke takes off it, each pixel's colour kept."""
    y0, y1, x0, x1 = box
    part = E8[y0:y1, x0:x1].astype(np.float32) - floor
    y = part @ C._LUMA.astype(np.float32)
    sharp = DETAIL[0] * (y - smoke_blur(y, smoke, DETAIL[1] * min(E8.shape[:2]) / 1080))
    gain = np.clip(1.0 + smoke * sharp / np.maximum(y, 2.0), 0.5, 2.0)
    E8[y0:y1, x0:x1] = np.clip(floor + part * gain[..., None] + 0.5, 0, 255).astype(np.uint8)
    return E8


def soot_billows(E8: np.ndarray, floor: int, box, ref: float, smoke: np.ndarray) -> np.ndarray:
    """Over ``box``: the render's own shading of its smoke (before fill_soot) as a factor less one, round ``ref`` (its
    smoke's median linear luma), eased into BILLOW's range (clipped, its folds were flat plateaus: a camouflage print),
    and the upper faces of its billows lit from above (TOP_LIGHT); nothing where it is fire (``smoke`` in [0, 1])."""
    y0, y1, x0, x1 = box
    e = C._black_floor_f(E8[y0:y1, x0:x1], floor)
    yl = cv2.GaussianBlur((C._lin(e) @ C._LUMA).astype(np.float32), (0, 0), 1.0)
    s = BILLOW[2] * np.log(np.maximum(yl, 1e-5) / max(ref, 1e-4))
    lo, hi = math.log(BILLOW[0]), math.log(BILLOW[1])
    s = np.where(s > 0, hi * np.tanh(s / hi), lo * np.tanh(s / lo))
    sigma = TOP_LIGHT[1] * min(E8.shape[:2])
    height = smoke_blur(np.log(np.maximum(yl, 1e-4)), smoke, sigma)  # (a billow is a bright lump: its height)
    up = np.tanh(np.maximum(np.gradient(height, axis=0) * sigma, 0.0) / 0.3)  # (it rises towards the lump's middle)
    return ((np.exp(s) * (1.0 + TOP_LIGHT[0] * up) - 1.0) * smoke).astype(np.float32)


def radial_blur(E8: np.ndarray, centre, amount: float, steps: int = 7) -> np.ndarray:
    """The fireball smeared outward from ``centre`` over the shutter: averaged over scales 1 .. 1 + ``amount`` (its
    growth this frame, times the shutter). A blast that grows a third in a frame is not crisp on a phone."""
    if amount < 0.02:
        return E8
    h, w = E8.shape[:2]
    acc = np.zeros(E8.shape, np.float32)
    for k in np.linspace(1.0, 1.0 + amount, steps):
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), 0.0, float(1.0 / k))
        acc += cv2.warpAffine(E8, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    return np.clip(acc / steps + 0.5, 0, 255).astype(np.uint8)


def rim_light(m: np.ndarray, src) -> np.ndarray:
    """(h, w) in [0, 1]: how much of the fire behind the arm wraps round its outline: a thin band just inside the
    matte's edge (RIM_BAND), on the side facing the fire at ``src`` (its outward normal on the direction to it,
    RIM_FACE)."""
    h, w = m.shape
    q = 2
    mq = cv2.resize(m, (w // q, h // q), interpolation=cv2.INTER_AREA)
    px = q * 1080.0 / min(h, w)  # a half-size px, in px at 1080
    d = cv2.distanceTransform((mq > 0.5).astype(np.uint8), cv2.DIST_L2, 5) * np.float32(px)
    gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 1.5))
    nrm = np.hypot(gx, gy) + 1e-6
    nx, ny = -gx / nrm, -gy / nrm  # outward
    yy, xx = np.indices(d.shape).astype(np.float32)
    lx, ly = src[0] / q - xx, src[1] / q - yy
    face = (nx * lx + ny * ly) / (np.hypot(lx, ly) + 1e-6)
    band = C._smoothstep(RIM_BAND[0] - 2.0, RIM_BAND[0] + 1.0, d) \
        * np.exp(-np.maximum(d - RIM_BAND[0], 0.0) / RIM_BAND[1])
    rim = band * C._smoothstep(RIM_FACE[0], RIM_FACE[1], face) * (mq > 0.5)
    return cv2.resize(rim.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def hand_coverage(plate8: np.ndarray, masks: np.ndarray | None, t: int) -> np.ndarray:
    """Solid current-frame skin inside the moving hand's local support. Adjacent mattes only locate the search;
    current plate colour decides coverage, so closing fingers cannot reveal fire and an old finger cannot fill a
    gap. The native blur's colour and texture are preserved; no RGB from an earlier hand pose is carried forward."""
    h, w = plate8.shape[:2]
    if masks is None:
        return np.zeros((h, w), np.float32)
    t = min(t, len(masks) - 1)
    m = masks[t].astype(np.float32) / 255.0
    support = np.max(masks[max(0, t - 1):min(len(masks), t + 2)], axis=0)
    support = cv2.dilate((support > 40).astype(np.uint8), np.ones((5, 5), np.uint8))
    radius = max(2, int(0.008 * min(h, w)))
    support &= cv2.dilate((m > 0.15).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                 (2 * radius + 1,) * 2))
    rgb = plate8.astype(np.float32)
    # A warm wall can share skin's red-minus-green value. Its missing blue makes
    # it yellow instead: rejecting that colour keeps original unlit wall out of
    # the foreground restoration. Smooth phone chroma blocks before the key so
    # they cannot turn the contour into isolated flecks and a jagged fringe.
    chroma = (rgb[..., 0] + rgb[..., 2] - 2.0 * rgb[..., 1]) / np.maximum(rgb.sum(-1), 1.0)
    chroma = cv2.GaussianBlur(chroma, (0, 0), max(0.6, min(h, w) / 1080.0))
    reliable = chroma[(m > 0.95) & (C._peak(rgb) > 50)]
    if reliable.size > 32:
        low = max(0.015, float(np.percentile(reliable, 10)) - 0.035)
        skin = C._smoothstep(low, low + 0.03, chroma) * support
        depth = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        core = C._smoothstep(2.0, max(3.0, 0.008 * min(h, w)), depth)
        # At the contour the segmenter's leftover wall pixels would restore an unlit, hairy border. Use the
        # current colour there; keep shaded and low-saturation skin safely opaque deeper inside the hand.
        m = np.maximum(m * np.maximum(core, skin), skin)
        m = cv2.GaussianBlur(m, (0, 0), 1.0)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ the fireball's light
def falloff(shape: tuple[int, int], r0: float, ball: float, power: float = 1.0) -> tuple[np.ndarray, float]:
    """The light's spread on a (h, w) grid: the rFFT of (1 + r / ``r0``) ** -``power`` over the grid doubled (so
    nothing wraps round the frame), and its sum over a disc of radius ``ball``: a ball that size of luminance 1
    lights its own middle 1."""
    h, w = shape
    dy = np.minimum(np.arange(2 * h), 2 * h - np.arange(2 * h)).astype(np.float64)[:, None]
    dx = np.minimum(np.arange(2 * w), 2 * w - np.arange(2 * w)).astype(np.float64)[None]
    r = np.hypot(dx, dy)
    k = (1.0 + r / max(r0, 0.5)) ** -power
    return np.fft.rfft2(k), float(k[r <= max(ball, 1.0)].sum())


def blast_light(y: np.ndarray, spread: np.ndarray, norm: float) -> np.ndarray:
    """(h, w): the light the glowing image ``y`` (linear luminance, on falloff's grid) throws round it."""
    h, w = y.shape
    lit = np.fft.irfft2(np.fft.rfft2(y.astype(np.float64), (2 * h, 2 * w)) * spread, (2 * h, 2 * w))[:h, :w]
    return np.maximum(lit / norm, 0.0).astype(np.float32)


def light_colour(fire_lin: np.ndarray) -> np.ndarray:
    """The colour (linear, luminance 1) of the light of ``fire_lin`` (the render's fire, linear): a camera's fire
    colour (RAMP) for its luminance-weighted brightness as the composite shows it, held to LIGHT_V. A hot ball lights
    the room yellow-orange, a cooling one deep orange."""
    y = fire_lin @ C._LUMA
    total = float(y.sum())
    v = LIGHT_V[1] if total <= 1e-9 else float(np.clip((float((y * y).sum()) / total * PARAMS.fire_gain) ** (1 / 2.2),
                                                       *LIGHT_V))
    c = np.array([np.interp(v, RAMP_V, RAMP[:, i]) for i in range(3)]) ** 2.2
    return (c / float(c @ C._LUMA)).astype(np.float32)


_STEPS = 8192  # the tables' steps: a unit of the square root of linear light (to 4), of display values (to 1)


@functools.lru_cache(maxsize=None)
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The shot's highlight shoulder (bending.SHOULDER, a tanh above it) as lookups, a pixel's arithmetic a lookup
    (computed per pixel it was a fifth of the frame's time): linear light (by its square root) to what the shoulder
    makes of it, linear and as display values; display values to linear back through it; and the shoulder's linear
    of each 8-bit plate value."""
    from .bending import SHOULDER as S

    def curve(x: np.ndarray) -> np.ndarray:
        return np.where(x > S, S + (1.0 - S) * np.tanh((x - S) / (1.0 - S)), x)

    x = (np.arange(2 * _STEPS + 1, dtype=np.float64) / _STEPS) ** 2
    y = np.minimum(np.arange(_STEPS + 1, dtype=np.float64) / _STEPS, 0.9999) ** 2.2
    back = np.where(y > S, S + (1.0 - S) * np.arctanh(np.clip((y - S) / (1.0 - S), 0.0, 0.9999)), y)
    return (curve(x).astype(np.float32), (curve(x) ** (1 / 2.2)).astype(np.float32), back.astype(np.float32),
            curve((np.arange(256) / 255.0) ** 2.2).astype(np.float32))


def _by_root(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``table`` (by the square root of linear light, _tables) looked up for each of ``x`` (linear, held to [0, 4])."""
    i = np.clip(x, 0.0, 4.0)
    np.sqrt(i, out=i)
    i *= np.float32(_STEPS)
    i += np.float32(0.5)
    return table.take(i.astype(np.int32))


def warm_light(P: np.ndarray, lit: np.ndarray, m: np.ndarray, rim: np.ndarray, burn: float = 0.0,
               response: float = 1.0, relief: float = 1.0) -> np.ndarray:
    """``lit`` (the room ``P`` with the fireball's light on it, linear) with what the light adds kept warm as the
    picture shows it (display values, after the shot's highlight shoulder, bending.SHOULDER). On the room: green rising
    at most LIGHT_WARM of red, blue not at all. On the arm (its matte ``m``): the skin's hue kept (green up to SKIN_GR
    of red, blue rising SKIN_BG of green) and its lift held to SKIN_LIFT, its face's to the first, its ``rim``'s to the
    second and, ``burn`` (BURN) of the way to a burning ball, the third more, its rim then going warm white
    (SKIN_RIM, and the band nearest its outline towards HOT_RIM's white). ``response`` is the fire's shared pulse,
    applied after limiting the highlight so saturation cannot turn flickering firelight into a flat plateau.
    ``relief`` restores native skin shading after the initial flash, keeping dark creases below lit rim faces."""
    _, shown, back, _ = _tables()
    g0 = _by_root(shown, P)
    d = _by_root(shown, lit)
    d -= g0
    b0, a0, bw, aw = cv2.boundingRect((m > 0).astype(np.uint8))
    arm = None
    if bw and aw:  # (the arm, only over the box round it)
        a1, b1 = a0 + aw, b0 + bw
        g, e, r, hot = g0[a0:a1, b0:b1], d[a0:a1, b0:b1], rim[a0:a1, b0:b1], rim[a0:a1, b0:b1] * np.float32(burn)
        red = np.maximum(e[..., 0], 0.0)
        green = np.clip((SKIN_GR + (SKIN_RIM[0] - SKIN_GR) * hot) * (g[..., 0] + red) - g[..., 1], 0.0,
                        np.maximum(e[..., 1], 0.0))
        skin = np.stack([red, green, np.clip(e[..., 2], 0.0, (SKIN_BG + (SKIN_RIM[1] - SKIN_BG) * hot) * green)], -1)
        shade = 1.0 + np.float32(relief) * (np.clip((g @ C._LUMA) / SKIN_SHADE, 0.0, 1.0) - 1.0)
        most = (SKIN_LIFT[0] + (SKIN_LIFT[1] + SKIN_LIFT[2] * burn - SKIN_LIFT[0]) * r) / 255.0
        most *= shade
        skin *= np.minimum(1.0, most / np.maximum(skin @ C._LUMA, 1e-6))[..., None]
        arm = (slice(a0, a1), slice(b0, b1)), skin
    np.minimum(d[..., 1], LIGHT_WARM * np.maximum(d[..., 0], 0.0), out=d[..., 1])  # (the room's)
    np.minimum(d[..., 2], 0.0, out=d[..., 2])
    if arm is not None:
        box, skin = arm
        d[box] += m[box][..., None] * (skin - d[box])
    d += g0
    if arm is not None and burn > 0:  # (its hot edge, as shown)
        box = arm[0]
        hot = np.float32(HOT_RIM[0] * burn) * C._smoothstep(*HOT_RIM[1], rim[box]) * m[box] * shade ** 2
        d[box] += hot[..., None] * np.maximum(np.array(HOT_RIM[2], np.float32) / 255.0 - d[box], 0.0)
    d = g0 + np.float32(response) * (d - g0)
    np.clip(d, 0.0, 1.0, out=d)
    d *= np.float32(_STEPS)
    d += np.float32(0.5)
    return back.take(d.astype(np.int32))  # (and back from the shoulder)


# ------------------------------------------------------------------ its outline
def _rank(a: np.ndarray) -> np.ndarray:
    """``a`` (float) as float32 uniform in [0, 1]: each value's rank."""
    rank = np.empty(a.size, np.float32)
    rank[np.argsort(a, axis=None)] = np.linspace(0.0, 1.0, a.size, dtype=np.float32)
    return rank.reshape(a.shape)


def _field(rng: np.random.Generator, h: int, w: int, across: float, up: float) -> np.ndarray:
    """(h, w) float32 uniform in [0, 1]: white noise blurred ``across`` and ``up`` px (by FFT, so it tiles)."""
    f = np.fft.rfft2(rng.standard_normal((h, w)))
    ky, kx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None]
    return _rank(np.fft.irfft2(f * np.exp(-2 * np.pi ** 2 * ((kx * across) ** 2 + (ky * up) ** 2)), (h, w)))


class Outline:
    """The smoke's outline broken up, frame by frame over the element's box: its outer lobes eaten away by crisp wisps
    fixed in the cloud (rising and curling with it), which breaks its round silhouette, the cloud thinning from its
    crown as it climbs and drawing in from its underside. Seeded per shot."""

    def __init__(self, h: int, w: int, side: int, seed: int):
        rng = np.random.default_rng([seed, 53])
        self.side = side
        hq, wq = max(2, h // 4), max(2, w // 4)
        self.wisps = [_field(rng, hq, wq, WISPS[0] * side / 4, WISPS[1] * side / 4) for _ in range(2)]
        for i, fine in enumerate(_field(rng, hq, wq, WISPS[0] * side / 12, WISPS[1] * side / 12) for _ in range(2)):
            self.wisps[i] = _rank(0.65 * self.wisps[i] + 0.35 * fine)  # (and their ragged edges)
        psi = _field(rng, hq, wq, WISP_CURL[1] * side / 4, WISP_CURL[1] * side / 4)
        gy, gx = np.gradient(cv2.GaussianBlur(psi, (0, 0), 2.0))
        top = float(np.percentile(np.hypot(gx, gy), 99)) + 1e-9
        self.curl = (np.clip(gy / top, -1, 1).astype(np.float32), np.clip(-gx / top, -1, 1).astype(np.float32))

    @staticmethod
    def _inside(part: np.ndarray, floor: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(element in [0, 1], its cloud's silhouette, signed px in from its edge), at ``size`` (w, h). The render's
        thin threads of haze round the cloud are not the cloud (kept as such they drew thin dark lines on the wall)."""
        e = C._black_floor_f(cv2.resize(part, size, interpolation=cv2.INTER_AREA), floor)
        inside = (C._smoothstep(0.03, 0.10, C._peak(e)) > 0.5).astype(np.uint8)
        inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        sd = cv2.distanceTransform(inside, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)
        return e, inside, sd

    def maps(self, E8: np.ndarray, floor: int, box, since: float, rise: float, fire_weight: np.ndarray | None = None):
        """What finish and density need over ``box`` of ``E8`` (uint8, plate size; quarter-size maps), or None for no
        cloud yet."""
        y0, y1, x0, x1 = box
        part = E8[y0:y1, x0:x1]
        hb, wb = part.shape[:2]
        hq, wq = max(2, hb // 4), max(2, wb // 4)
        e, inside, sd = self._inside(part, floor, (wq, hq))
        if inside.sum() < 12:
            return None
        rc = math.sqrt(float(inside.sum()) / math.pi)  # its radius, quarter px
        cy = float(np.nonzero(inside)[0].mean())  # its middle's row
        yy, xx = np.mgrid[:hq, :wq].astype(np.float32)
        fire = np.clip(C.fire_weight_map(e, PARAMS, max(1, E8.shape[1] * wq // wb), True), 0.0, 1.0)
        held = fire if fire_weight is None else cv2.resize(fire_weight, (wq, hq), interpolation=cv2.INTER_AREA)
        near = np.clip(2.0 * cv2.GaussianBlur(held, (0, 0), max(1.0, 0.1 * rc)), 0.0, 1.0)  # (its dim rim too)
        # the wisps: fixed in the cloud as it rises (the second peeling off upward faster), curling as it lives
        swirl = np.float32(WISP_CURL[0] * self.side / 4 * C._smoothstep(0.0, WISP_CURL[2], since))
        wisp = np.zeros((hq, wq), np.float32)
        for field, share, faster in zip(self.wisps, (0.6, 0.4), (1.0, 1.5)):
            fxq = ((xx * (wb / wq) + x0) / 4).astype(np.float32)
            fyq = ((yy * (hb / hq) + y0 + faster * rise) / 4).astype(np.float32)
            cu = cv2.remap(self.curl[0], fxq, fyq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            cv = cv2.remap(self.curl[1], fxq, fyq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            wisp += share * cv2.remap(field, fxq + swirl * cu, fyq + swirl * cv, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_WRAP)
        wisp = np.clip(0.5 + 1.39 * (wisp - 0.5), 0.0, 1.0)  # (the two fields' mix spread back over [0, 1])
        lum = C._peak(e)  # (its billows: the render's light on them, over that of its brightest tenth)
        lum = np.clip(lum / max(float(np.percentile(lum[inside > 0], 90)), 1e-3), 0.0, 1.0)
        wisp = WISPS[3] * lum + (1.0 - WISPS[3]) * wisp
        outer = np.clip(1.0 - sd / (OUTER[0] * rc), 0.0, 1.0)  # (1 at its edge, 0 at the outer lobes' inner side)
        under = np.clip((yy - cy) / rc, 0, 1) * (1.0 - C._smoothstep(0.0, 0.6 * rc, sd))  # (its underside, from out)
        eat = OUTER[1] * outer ** OUTER[2] + (1.0 - C._smoothstep(-0.15 * rc, 0.0, sd)) \
            + BOTTOM_THIN[2] * C._smoothstep(BOTTOM_THIN[0], BOTTOM_THIN[1], since) * under  # (its haze past it goes)
        thin = 1.0 - TOP_THIN[2] * C._smoothstep(TOP_THIN[0], TOP_THIN[1], since) * np.clip((cy - yy) / rc, 0, 1)
        thin *= 1.0 - 0.6 * C._smoothstep(25.0, 65.0, since) * C._smoothstep(0.42, 0.7, wisp)
        # Only current fire bypasses wisp erosion. Holding that exemption with the material key preserves a flat
        # smoke skirt after its flame has gone; the fire layer itself supplies the gradual cooling handoff.
        own_fire = np.clip(2.0 * cv2.GaussianBlur(fire, (0, 0), max(1.0, 0.04 * rc)), 0.0, 1.0)  # (the fire: its
        # soot round it is eaten too, or it hung by the ball as a blurred dark crescent, a drop shadow)
        lobe = LOBE[0] * (LOBE[1] + (1.0 - LOBE[1]) * C._smoothstep(LOBE[2], LOBE[3], since))
        return near, wisp, eat, thin, (1.0 - near) * outer, own_fire, lobe

    @staticmethod
    def density(maps, size: tuple[int, int], side: int) -> np.ndarray:
        """(h, w) at ``size`` (w, h): how much of the composite's soot each pixel keeps: its outer lobes cut sharply at
        full size, thickening in from each cut edge (LOBE), the fire whole."""
        _, wisp, eat, thin, _, own, lobe = maps
        up = lambda a, how=cv2.INTER_LINEAR: cv2.resize(a.astype(np.float32), size, interpolation=how)  # noqa: E731
        t = up(eat)
        kept = C._smoothstep(t - WISPS[2], t + WISPS[2], up(wisp, cv2.INTER_CUBIC))
        depth = cv2.distanceTransform((kept > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        kept *= np.sqrt(np.clip((depth + 1.0) / (lobe * side), 0.0, 1.0)) * up(thin)
        return 1.0 - (1.0 - up(own)) * (1.0 - kept)

    def finish(self, Q: np.ndarray, P: np.ndarray, box, maps) -> None:
        """In place over ``box``: the composite ``Q`` (linear) with maps' outline put on it against the room ``P``:
        soot never lighter than the room behind it, a little softened in its outer lobes (the render draws a thin
        bright rim round its smoke: composited it drew a contour line round the cloud), its density where it is eaten
        and thinned."""
        near, _, _, _, outer, _, _ = maps
        y0, y1, x0, x1 = box
        size = (x1 - x0, y1 - y0)
        up = lambda a: cv2.resize(a.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)[..., None]  # noqa: E731
        q, p = Q[y0:y1, x0:x1], P[y0:y1, x0:x1]
        base = np.maximum(p, 0.02)
        rel = (q - p - (1.0 - up(near)) * np.maximum(q - p, 0.0)) / base  # (what it does to the room, as a share
        # of the room: softened as a difference, the dark it laid on a white switch plate spread black onto the wall)
        soft = up(outer)
        rel = rel + soft * (cv2.GaussianBlur(rel, (0, 0), OUTER_SOFT * self.side / 1080) - rel)
        Q[y0:y1, x0:x1] = np.maximum(p + base * rel * self.density(maps, size, self.side)[..., None], 0.0)


class SmokeVolume:
    """Photographed cooling soot hands over to seeded, self-shadowed turbulent smoke at several depths."""

    def __init__(self, h: int, w: int, side: int, seed: int):
        self.side, self.seed = side, seed
        rng = np.random.default_rng([seed, 137])
        self.fields = [_field(rng, max(2, h // 4), max(2, w // 4), side * a / 4, side * b / 4)
                       for a, b in ((0.055, 0.075), (0.032, 0.055))]
        self.h, self.w = h, w
        self.volume_noise = None

    def _prepare_noise(self):
        if self.volume_noise is not None:
            return
        px = self.side / 1080.0
        q = max(1, round(3 * px))
        shape = (40, math.ceil(self.h / q), math.ceil(self.w / q))
        # One fixed 3D field at three scales. Its coordinates rise continuously with the cloud.
        rng = np.random.default_rng([self.seed, 317])
        ft = np.fft.rfftn(rng.standard_normal(shape))
        kz = np.fft.fftfreq(shape[0])[:, None, None]
        ky = np.fft.fftfreq(shape[1])[None, :, None]
        kx = np.fft.rfftfreq(shape[2])[None, None, :]
        self.volume_noise = []
        for level in (1.0, 2.8, 5.2):
            sigma = level * 8 * px / q
            spectrum = ft * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (0.25 * kz ** 2 + ky ** 2 + kx ** 2))
            field = np.fft.irfftn(spectrum, shape, axes=(0, 1, 2)).astype(np.float32)
            field /= max(float(field.std()), 1e-6)
            self.volume_noise.append(field)
        del ft, spectrum, field, kz, ky, kx

    def draw(self, room, source, silhouette, floor, ambient, since, rise, clear, box, erosion=None):
        """Keep the cooling source intact, then resolve late soot through real overlapping depths."""
        if clear <= 0.0:
            return room.copy()
        mix = float(C._smoothstep(*LOBE_SEPARATE, since))
        if mix <= 0.0:
            return self._cooling(room, source, silhouette, floor, ambient, since, rise, clear, box, erosion)
        result = self._draw_volume(room, source, silhouette, floor, ambient, since, rise, clear, box, erosion)
        if mix < 1.0:
            previous = self._cooling(room, source, silhouette, floor, ambient, since, rise, clear, box, erosion)
            result = previous + np.float32(mix) * (result - previous)
        return result

    def _draw_volume(self, room, source, silhouette, floor, ambient, since, rise, clear, box, erosion=None):
        """Integrate seeded turbulent soot and room light through five unequal bodies, retaining source wrinkles."""
        hb, wb = source.shape[:2]
        y0, _, x0, _ = box
        px = self.side / 1080.0
        yy, xx = np.mgrid[:hb, :wb].astype(np.float32)
        peak = np.maximum(C._peak(source).astype(np.float32) - floor, 0.0) / (255 - floor)
        geometry = np.maximum(C._smoothstep(0.012, 0.085, peak), 0.92 * silhouette)
        key = geometry if erosion is None else geometry * erosion
        total = float(key.sum())
        if total <= 0.01:
            return room.copy()
        cy, cx = float((key * yy).sum() / total), float((key * xx).sum() / total)
        radius = max(1.0, math.sqrt(total / math.pi))
        # Resolve the dense boundaries at roughly 360 samples across a 1080-wide plate; source wrinkles stay native.
        q = max(1, round(3 * px))
        ny, nx, nz = math.ceil(hb / q), math.ceil(wb / q), 40
        self._prepare_noise()
        yl, xl = np.mgrid[:ny, :nx].astype(np.float32)
        yn, xn = yl * q, xl * q
        mx = (xl + x0 / q + 0.10 * rise / q).astype(np.float32)
        my = (yl + y0 / q + 1.2 * rise / q).astype(np.float32)
        noise = [np.stack([cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) for a in field])
                 for field in self.volume_noise]
        turbulent = 0.40 * noise[0] + 0.60 * noise[1] + 0.65 * noise[2]
        del noise
        z = np.linspace(-1.1, 1.1, nz, dtype=np.float32)[:, None, None]
        density = np.zeros((nz, ny, nx), np.float32)
        pigment, strongest = np.zeros_like(density), np.zeros_like(density)
        # Position and unequal depths, radii and albedos: rear shoulders then two denser foreground bodies.
        lobes = [(-0.44, -0.62, -0.25, 0.47, 0.51, 0.48, 0.434),
                 (0.14, -0.72, 0.18, 0.54, 0.52, 0.47, 0.448),
                 (0.54, -0.20, 0.10, 0.41, 0.40, 0.44, 0.378),
                 (-0.45, -0.35, -0.20, 0.37, 0.22, 0.42, 0.36),
                 (0.20, -0.15, 0.72, 0.42, 0.34, 0.30, 0.28)]
        for i, (dx, dy, dz, rx, ry, rz, albedo) in enumerate(lobes):
            rx, ry = rx * 1.05, ry * 1.05
            u = (xn - (cx + dx * radius)) / (rx * radius)
            v = (yn - (cy + dy * radius)) / (ry * radius)
            r2 = u[None] ** 2 + v[None] ** 2 + ((z - dz) / rz) ** 2
            local = np.roll(turbulent, (i * 7, i * 9, i * 13), axis=(0, 1, 2))
            path = np.maximum(1.05 - r2 + 0.30 * local, 0.0)
            # Dilute rear shoulders have smaller opaque cores and thin boundaries; the near core overlaps them.
            den = np.minimum(path, 1.4) ** (1.7 if i < 2 else 1.5)
            if i < 2:
                den *= 0.75
            if i == 3:  # The lighter shoulder disperses before the lower foreground core, breaking a paired skirt.
                den *= 1.0 - 0.6 * float(C._smoothstep(30.0, 60.0, since))
            if i == 4:  # Release the hanging foreground skirt before the upper cores; keep its denser fragments.
                release = float(C._smoothstep(30.0, 48.0, since))
                kernel = C._smoothstep(0.65, 1.15, den)
                den *= 1.0 - 0.65 * release * (1.0 - 0.80 * kernel)
            density += den
            pigment += den * albedo
            strongest = np.maximum(strongest, den)
        albedo = pigment / np.maximum(density, 1e-5)
        del pigment, u, v, r2, local, path, den, turbulent
        # Open the connections in the lower body, retaining individual dense kernels rather than fading all soot.
        lower = C._smoothstep(cy - 0.25 * radius, cy + 0.10 * radius, yn)[None]
        kernels = C._smoothstep(0.55, 1.15, strongest)
        density *= 1.0 - 0.30 * lower * (1.0 - kernels)
        del strongest, kernels
        overhead = np.cumsum(density, axis=1) * np.float32(0.032 * q / (8 * px))
        overhead = np.stack([cv2.GaussianBlur(a, (0, 0), max(0.5, 3.5 * px / q)) for a in overhead])
        frontshade = np.cumsum(density, axis=0) * np.float32(0.025)
        lighting = 0.40 + 0.60 * np.exp(-overhead - 0.5 * frontshade)
        del overhead, frontshade
        # Compact fold faces catch overhead room light; diffuse fill bounds their shadow contrast and preserves
        # their ambient response. Weak gradients in dilute smoke leave the separate wisps soft.
        slope = np.tanh(np.gradient(density, axis=1) * (8 * px / q))
        lighting *= 1.12 * (1.0 + 1.2 * np.maximum(slope, -0.2))
        del slope
        # Dilute voxels clear first; deep kernels retain their path instead of the entire cloud fading together.
        tau = density * np.float32(0.20) * np.float32(clear) ** (0.8 + 1.4 * (1.0 - C._smoothstep(0.3, 1.0, density)))
        del density
        transmission = np.ones((ny, nx), np.float32)
        scattered = np.zeros((ny, nx), np.float32)
        for iz in range(nz):
            trans = np.exp(-tau[iz])
            scattered = scattered * trans + albedo[iz] * lighting[iz] * (1.0 - trans)
            transmission *= trans
        del tau, albedo, lighting
        trans = np.clip(cv2.resize(transmission, (wb, hb), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
        smoke = np.maximum(cv2.resize(scattered, (wb, hb), interpolation=cv2.INTER_CUBIC), 0.0)
        # Source relief adds photographic wrinkles to the lighting and bounded variation to the soot path.
        grey = cv2.GaussianBlur(peak, (0, 0), max(0.65, 1.5 * px))
        log = np.log(np.maximum(grey, 0.008))
        fine = log - smoke_blur(log, geometry, max(1.0, 22 * px))
        detail = np.clip(fine, -0.8, 0.8)
        present = geometry > 0.2
        reference = max(float(np.percentile(grey[present], 65)), 0.025) if present.any() else 0.025
        normalized = log - math.log(reference)
        # Unresolved photographic clefts also contain longer soot paths when ambient illumination is zero.
        path_factor = np.clip(np.exp(-0.25 * np.tanh(normalized) - 0.35 * detail), 0.75, 1.35)
        folded_trans = trans ** path_factor
        smoke *= (1.0 - folded_trans) / np.maximum(1.0 - trans, 1e-5)
        trans = folded_trans
        slope = np.tanh(np.gradient(smoke_blur(normalized, geometry, max(1.0, 10 * px)), axis=0) * 10 * px)
        photo = np.clip(np.exp(0.20 * np.tanh(normalized) + 1.15 * detail), 0.35, 2.0) * (1.0 + 0.40 * slope)
        smoke *= photo
        # The detached thin smoke follows a faster field and clears ahead of the compact bodies. Put it behind
        # their depths so opening a channel never restores the old opaque arch across the foreground lobes.
        wisps = cv2.remap(self.fields[0], (xx + x0 + 0.22 * rise).astype(np.float32) / 4,
                          (yy + y0 + 2.1 * rise).astype(np.float32) / 4,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wisp_tau = np.float32(0.12 * clear ** 2.6) * key * C._smoothstep(0.3, 0.75, wisps)
        wisp_trans = np.exp(-wisp_tau)
        smoke += trans * np.float32(SOOT_ALBEDO) * (1.0 - wisp_trans)
        trans *= wisp_trans
        # A volume may open gaps inside the source support, but cannot paint a new footprint outside it.
        trans = 1.0 + geometry * (trans - 1.0)
        smoke *= geometry
        light = wide_blur(ambient, max(1.0, 0.035 * self.side))
        wall = wide_blur(room, max(1.0, 0.08 * self.side))
        result = room * trans[..., None] + light * smoke[..., None]
        result += (wall - room) * (trans * (1.0 - trans ** 2.5))[..., None]
        display = np.maximum(room, 0.0) ** np.float32(1 / 2.2)
        residual = display - cv2.GaussianBlur(display, (0, 0), max(0.5, 0.7 * px))
        grain = float(np.clip(1.4826 * np.median(np.abs(residual)), 0.35 / 255, 1.5 / 255))
        n = np.random.default_rng([self.seed, 173, int(since)]).standard_normal((hb, wb)).astype(np.float32)
        result += (2.2 * grain * np.sqrt(np.maximum(1.0 - trans ** 7, 0.0)) * n)[..., None] \
            * np.maximum(result, 1e-5) ** np.float32(1 - 1 / 2.2)
        return np.maximum(result, 0.0)

    def _cooling(self, room: np.ndarray, source: np.ndarray, silhouette: np.ndarray, floor: int,
             ambient: np.ndarray, since: float, rise: float, clear: float, box,
             erosion: np.ndarray | None = None) -> np.ndarray:
        """The cloud over this room crop, in linear light. Detail transmitted through soot is attenuated more than
        the room's broad light; no fitted bright background object can become part of the smoke's albedo. Lighting
        comes only from ``ambient`` for this frame, including the decaying fire, with no retained orange tint."""
        if clear <= 0.0:
            return room.copy()
        hb, wb = source.shape[:2]
        y0, _, x0, _ = box
        px = self.side / 1080.0
        yy, xx = np.mgrid[:hb, :wb].astype(np.float32)
        peak = np.maximum(C._peak(source).astype(np.float32) - floor, 0.0) / (255 - floor)
        geometry = np.maximum(C._smoothstep(0.012, 0.085, peak), 0.92 * silhouette)
        support = geometry.copy()
        if erosion is not None:
            support *= erosion
        present = support > 0.005
        if not present.any():
            return room.copy()
        cy = float((support * yy).sum() / support.sum())
        radius = max(1.0, math.sqrt(float(support.sum()) / math.pi))
        # Texture is measured before the old dense fill. Its bright ridges and dark clefts remain independent of
        # opacity, so self-shadowed folds obscure the wall instead of opening pale holes through the cloud.
        grey = cv2.GaussianBlur(peak, (0, 0), max(0.65, 1.5 * px))
        grey = np.maximum(grey + 0.6 * (grey - smoke_blur(grey, support, max(1.0, 6.0 * px))), 0.0)
        ref = max(float(np.percentile(grey[present], 65)), 0.025)
        relief = np.log(np.maximum(grey, 0.008) / ref)
        broad = smoke_blur(relief, support, max(1.0, 9.0 * px))
        slope = np.tanh(np.gradient(broad, axis=0) * max(1.0, 12.0 * px))
        shade = np.exp(0.65 * np.tanh(relief)) * (1.0 + 0.48 * slope)
        shade = np.clip(shade, 0.3, 2.2).astype(np.float32)
        wall = wide_blur(room, max(1.0, 0.08 * self.side))
        # Broad room light, rather than plate pixels underneath each lobe, illuminates airborne soot.
        light = wide_blur(ambient, max(1.0, 0.035 * self.side))
        q = room.copy()
        remaining = np.ones((hb, wb), np.float32)
        for index, field in enumerate(self.fields):
            speed = 0.35 + 0.65 * index
            mx = xx + np.float32(index * 0.012 * self.side * math.sin(since / 21.0))
            my = yy + np.float32(index * 0.18 * rise)
            noise = cv2.remap(field, (xx + np.float32(x0 + speed * 0.12 * rise)) / 4,
                              (yy + np.float32(y0 + (1.0 + speed) * rise)) / 4,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            cov = cv2.remap(support, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            depth = cv2.distanceTransform((cov > 0.45).astype(np.uint8), cv2.DIST_L2, 3)
            edge = 1.0 - C._smoothstep(0.0, max(2.0, 0.075 * self.side), depth)
            # A smooth optical taper leaves both translucent strands and soft gaps. Its fine structure moves with
            # the layer, while the photographic billows retain their own native edge softness.
            porous = C._smoothstep(0.24 + 0.34 * edge, 0.82, noise)
            # Weak paths must remain weak through both overlapping layers. A broad density floor sealed them
            # into one opaque patch; sparse paths and thinner fringes retain the dense photographed folds.
            density = cov * (0.08 * (1.0 - edge) + (0.92 + 0.08 * edge) * porous ** 1.5) * (1.0 - 0.8 * edge)
            if index == 1:
                density *= C._smoothstep(0.3, 0.68, noise)
            # Broad lower skirts disperse ahead of compact cores. Keep the denser foreground folds while removing
            # the suspended apron that otherwise survives under the rising cloud after the last visible flame.
            underside = np.clip((my - cy) / radius, 0.0, 1.0)
            density *= 1.0 - 0.2 * C._smoothstep(25.0, 38.0, since) * underside \
                * (1.0 - 0.75 * C._smoothstep(0.55, 0.8, noise))
            density *= np.float32(clear) ** (1.3 + 1.6 * edge + 0.7 * (1.0 - noise))
            outline_depth = cv2.distanceTransform((cv2.remap(geometry, mx, my, cv2.INTER_LINEAR,
                                                            borderMode=cv2.BORDER_CONSTANT) > 0.45).astype(np.uint8),
                                                  cv2.DIST_L2, 3)
            interior = C._smoothstep(0.0, max(3.0, 0.025 * self.side), outline_depth)
            local_relief = cv2.remap(relief, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # Self-shadowed folds contain a longer path through soot. Keeping this depth in the photographed
            # interior, independently of its colour, restores folds even after the fire's illumination has died.
            density *= 1.0 + interior * (np.clip(np.exp(-0.45 * local_relief), 0.7, 1.5) - 1.0)
            tau = np.float32(VOLUME_DEPTH * (0.6 if index == 0 else 1.0)) * density
            trans = np.exp(-tau)
            local_shade = cv2.remap(shade, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # The source's fade into black describes coverage, not a groove around every lobe. Relief belongs
            # inside the volume; carrying it to the alpha boundary etched a double contour around the cloud.
            local_shade = 1.0 + (local_shade - 1.0) * interior
            if index == 1:
                front = cv2.GaussianBlur(density, (0, 0), max(1.0, 0.015 * self.side))
                local_shade *= 1.0 + 0.35 * np.tanh(np.gradient(front, axis=0) * 0.035 * self.side)
            colour = light * (SOOT_ALBEDO * local_shade)[..., None]
            # Multiple scattering removes high-contrast background detail early. The broad wall illumination is
            # still transmitted, with more detail visible in genuinely thin wisps and clear gaps untouched.
            q = q * trans[..., None] + colour * (1.0 - trans)[..., None]
            remaining *= trans
        q += (wall - room) * (remaining * (1.0 - remaining ** 2.5))[..., None]
        # Match the plate's robust fine-noise level, without copying its edges into opaque smoke. Directly adding
        # a plate high-pass back here would redraw the switch's border after transmission had hidden it.
        display = np.maximum(room, 0.0) ** np.float32(1 / 2.2)
        residual = display - cv2.GaussianBlur(display, (0, 0), max(0.5, 0.7 * px))
        grain = float(np.clip(1.4826 * np.median(np.abs(residual)), 0.35 / 255, 1.5 / 255))
        noise = np.random.default_rng([self.seed, 173, int(since)]).standard_normal((hb, wb)).astype(np.float32)
        # Fine background detail transmits by remaining**3.5 above. Restore its lost noise variance, rather than
        # scaling noise by opacity: the latter made translucent smoke conspicuously smoother than the phone plate.
        q += (2.2 * grain * np.sqrt(np.maximum(1.0 - remaining ** 7, 0.0)) * noise)[..., None] \
            * np.maximum(q, 1e-5) ** np.float32(1 - 1 / 2.2)
        return np.maximum(q, 0.0)


def scatter(Q: np.ndarray, P: np.ndarray, box, side: int) -> None:
    """In place over ``box``: the soot of the composite ``Q`` (linear) over the room ``P`` as smoke that scatters the
    room's light as well as hiding what is behind it (SCATTER). Locally (over windows SCATTER[1] of the short side) the
    composite's luminance is a share ``a`` of the room's behind it plus its soot's own light (a linear fit, as a
    guided filter makes it); where it leaves a share f of the room, what is behind now shows through ``a`` f **
    SCATTER[0], softened as the soot thickens, and the light it no longer lets through is the room's light round it
    (blurred SCATTER[3] of the short side) scattered back, so a white switch plate goes under the cloud as the wall
    beside it does. Its colour is that light's, greyed (SOOT_HUE). Where it is no darker than the room (its fire) it is
    as it was. (Its fit, light and colour are smooth: made at half size.)"""
    y0, y1, x0, x1 = box
    q, p = Q[y0:y1, x0:x1], P[y0:y1, x0:x1]
    ql, pl = q @ C._LUMA, p @ C._LUMA
    soot = C._smoothstep(0.98, 0.85, ql / np.maximum(pl, 1e-4))
    if float(soot.max()) <= 0.0:
        return
    hb, wb = ql.shape
    half = lambda a: cv2.resize(a, (max(2, wb // 2), max(2, hb // 2)), interpolation=cv2.INTER_AREA)  # noqa: E731
    full = lambda a: cv2.resize(a, (wb, hb), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    p2, q2 = half(p), half(ql)
    l2 = p2 @ C._LUMA
    k = max(3, int(SCATTER[1] * side / 2) | 1)
    mean = lambda a: cv2.boxFilter(a, -1, (k, k), borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mp, mq = mean(l2), mean(q2)
    f = np.clip(mq / np.maximum(mp, 1e-4), 0.0, 1.0)
    a = mean(np.clip((mean(l2 * q2) - mp * mq) / (mean(l2 * l2) - mp * mp + np.float32(SCATTER[4])), 0.0, 1.0))
    t = f ** np.float32(SCATTER[0])
    blurred, lit = wide_blur(p2, SCATTER[2] * side / 2160), wide_blur(p2, SCATTER[3] * side / 2)
    local = wide_blur(p2, 0.055 * side / 2)  # colour from the nearby wall, not a different surface across the room
    light = t[..., None] * (p2 + (1.0 - t)[..., None] * (blurred - p2)) + (1.0 - t)[..., None] * local
    grey = (light @ C._LUMA)[..., None]
    hue = full(1.0 + np.float32(SOOT_HUE) * (light / np.maximum(grey, 1e-5) - 1.0))
    a, t = full(a), full(t)
    soft = pl + (1.0 - t) * (full(blurred @ C._LUMA) - pl)
    ql2 = np.maximum(ql + a * (t * soft + (1.0 - t) * full(lit @ C._LUMA) - pl), 0.25 * ql)
    # The fitted density is spatially averaged, so it must not extend the scatter footprint beyond the soot that
    # actually obscures this pixel. Extending it recoloured clear wall into a pale blue-grey veil.
    q += soot[..., None] * (ql2[..., None] * hue - q)


_RAMP_STEPS = 1024
_FIRE_RAMP = RAMP.copy()
_FIRE_RAMP[1:3] = np.array([[70, 38, 22], [150, 72, 30]]) / 255.0  # (its dim end brown: a fire's dim edge is the
# smoke round it lit by it; deep red, every ember in the soot was ringed red)
_RAMP_LIN = (np.stack([np.interp(np.linspace(0.0, 1.0, _RAMP_STEPS + 1), RAMP_V, _FIRE_RAMP[:, c]) for c in range(3)],
                      -1) ** 2.2).astype(np.float32)  # in linear light, by display brightness


def ramp_lin(v: np.ndarray) -> np.ndarray:
    """A camera's fire colour (RAMP, its dim end brown, linear) for each display brightness in ``v``."""
    return _RAMP_LIN[(np.clip(v, 0.0, 1.0) * _RAMP_STEPS + 0.5).astype(np.int32)]


class Fire:
    """The fireball's fire drawn as a light over the composite (which reads the render's pale yellow body as fire-lit
    smoke and relights it as soot), frame by frame over the element's box: its brightness exposed as a phone exposed
    for the room shows it (EXPOSE, on RAMP), its highlights spread by the lens (BLOOM). While it burns (BURN) the ball
    keeps an opaque core (BALL) with translucent turbulent margins (FLAME_EDGE), its limb never darker than the wall
    round it (LIMB, WALL_CAP), nothing round it but the wall it lights; broad billows keep the phone's resolution. As
    it dies it stays one body of dimming fire over its own brown-black soot (BODY), its embers drawn
    over the soot round them (never ringed darker) and glowing into it (EMBER_GLOW), and its soot takes the render's own
    billows (SOOT_BANDS)."""

    def __init__(self, side: int, seed: int = 0):
        self.side, self.px, self.seed = side, side / 1080.0, seed
        self.previous_material = None
        rng = np.random.default_rng([seed, 191])
        self.edge_field = 0.7 * _field(rng, 128, 128, 7.0, 16.0) + 0.3 * _field(rng, 128, 128, 0.9, 2.0)

    def fringe(self, alpha, inner, mid, radius, since, strength):
        """Thin substantial boundary sections into advecting paths, keeping the inner 73% of the radius opaque."""
        yy, xx = np.mgrid[:alpha.shape[0], :alpha.shape[1]].astype(np.float32)
        scale = min(4.0 * self.px, radius / 60.0)
        reach = min(FLAME_EDGE[0] * self.px, 0.27 * radius)
        mx = (xx - mid[0] + 0.7 * since * self.px) / scale
        my = (yy - mid[1] + 1.6 * since * self.px) / scale
        noise = cv2.remap(self.edge_field, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        depth = inner + np.float32(min(FLAME_EDGE[1] * self.px, 0.55 * radius)) * (noise - 0.5)
        paths = C._smoothstep(-4.0 * self.px, 0.5 * reach, depth)
        paths *= C._smoothstep(0.18, 0.70, noise)
        paths *= C._smoothstep(-4.0 * self.px, 2.0 * self.px, inner)
        margin = 1.0 - C._smoothstep(0.10 * reach, reach, inner)
        return alpha + np.float32(strength) * margin * (paths - alpha)

    def material(self, wf: np.ndarray, box, centre, burn: float, cloud: np.ndarray | None = None) -> np.ndarray:
        """Fire-to-soot classification follows the cloud and releases over MATERIAL_RELEASE's interval. The source's sharp colour
        and billow texture stay current; only the material weight persists, so its last dark folds cannot become
        grey holes in one frame when saturation crosses the fire key's threshold. The placed cloud's soft support
        bounds that history, so a shrinking or deforming edge cannot leave a detached stain."""
        weight = np.clip(cv2.GaussianBlur(wf, (0, 0), 1.5 * self.px), 0.0, 1.0)
        if self.previous_material is not None and burn < 0.95:
            old, old_box, old_centre = self.previous_material
            dx = old_box[2] - box[2] + centre[0] - old_centre[0]
            dy = old_box[0] - box[0] + centre[1] - old_centre[1]
            carried = cv2.warpAffine(old, np.float32([[1, 0, dx], [0, 1, dy]]),
                                     (weight.shape[1], weight.shape[0]), flags=cv2.INTER_LINEAR)
            if cloud is not None:
                carried *= cv2.GaussianBlur(cloud, (0, 0), 1.5 * self.px)
            weight = np.maximum(weight, carried - MATERIAL_RELEASE)
        self.previous_material = weight, box, centre
        return weight

    def ball(self, wf: np.ndarray, burn: float = 1.0, cloud: np.ndarray | None = None):
        """The ball in the fire weight ``wf`` (box size), ``burn`` (BURN) of the way to burning: (its alpha, px in from
        its edge, both box size; its middle (x, y) and radius in box px), or None for no fire. As it dies, gaps in it
        narrower than BODY[3] of its radius are closed where the render has its ``cloud`` (box size, bool)."""
        hb, wb = wf.shape
        h2, w2 = max(2, hb // 2), max(2, wb // 2)
        own = (cv2.resize(wf, (w2, h2), interpolation=cv2.INTER_AREA) > BALL[0]).astype(np.uint8)
        own = cv2.morphologyEx(own, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(own, connectivity=8)
        if n < 2:
            return None
        area = st[1:, cv2.CC_STAT_AREA]
        own = np.isin(lab, 1 + np.nonzero(area >= 0.1 * area.max())[0]).astype(np.uint8)  # (not its stray sparks)
        r = int(BODY[3] * (1.0 - burn) * math.sqrt(float(own.sum()) / math.pi) / 4)  # (half the gap, quarter px)
        if cloud is not None and r >= 2:  # (the dying fire's own smoke in its folds open to the outside is its body)
            q4 = cv2.resize(own, (max(2, w2 // 2), max(2, h2 // 2)), interpolation=cv2.INTER_NEAREST)
            shut = cv2.morphologyEx(q4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
            shut = cv2.resize(shut, (w2, h2), interpolation=cv2.INTER_NEAREST)
            own |= shut & cv2.resize(cloud.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST)
            k = max(3, int(BODY[4] * self.px) | 1)  # (its thin strands are not its body: dim, drawn over its soot,
            # they were dark cracks)
            own = cv2.morphologyEx(own, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        own = cv2.drawContours(np.zeros_like(own), cs, -1, 1, -1)  # (its dark folds are inside it)
        sd2 = cv2.distanceTransform(own, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - own, cv2.DIST_L2, 5)
        sd2 = cv2.GaussianBlur(sd2, (0, 0), 1.0) * (wb / w2) - BALL[1] * self.px  # (box px in from its edge)
        inner = cv2.resize(sd2, (wb, hb), interpolation=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(own)
        mid = (float(xs.mean()) * wb / w2, float(ys.mean()) * hb / h2)
        soft = (BALL[2] + BODY[2] * (1.0 - burn)) * self.px
        return (np.clip(inner / soft + 0.5, 0.0, 1.0), inner, mid, math.sqrt(float(own.sum()) / math.pi) * wb / w2)

    @staticmethod
    def exposed(ef8: np.ndarray, floor: int) -> tuple[np.ndarray, np.ndarray]:
        """(the render's display luma, the fire's brightness as the phone shows it, EXPOSE) over its box ``ef8``."""
        v = np.clip(C._black_floor_f(ef8, floor) @ C._LUMA, 0.0, 1.0)
        w = 1.0 - (1.0 - v) ** np.float32(EXPOSE[0])
        return v, w ** (1.0 + np.float32(EXPOSE[1]) * (1.0 - w))

    def limb(self, u: np.ndarray, inner: np.ndarray, burn: float, A=1.0) -> np.ndarray:
        """The fire's brightness ``u`` in the burning ball (``A``: its alpha; ``inner``: px in from its edge), ``burn``
        (BURN) of the way to burning: its folds lifted towards white (EXPOSE[2]); at its limb, its folds filled up to
        the fire round them, never past it, and its darkest folds there to orange (LIMB, a smooth maximum)."""
        u = 1.0 - (1.0 - u) * (1.0 - np.float32(EXPOSE[2] * burn) * A)
        near = np.float32(burn) * (1.0 - C._smoothstep(LIMB[1] * self.px, LIMB[2] * self.px, inner))
        h, w = u.shape
        inside = (inner > 0).astype(np.float32)  # (the fire round it in the ball, not the black past its edge)
        half = lambda a: cv2.GaussianBlur(cv2.resize(a, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA),
                                          (0, 0), LIMB[0] * self.px / 2)  # noqa: E731
        round_it = cv2.resize(half(u * inside) / np.maximum(half(inside), 0.05), (w, h), interpolation=cv2.INTER_LINEAR)
        u = u + near * np.maximum(round_it - u, 0.0)
        least = np.float32(LIMB[3])
        return u + near * 0.5 * (least - u + np.sqrt((least - u) ** 2 + np.float32(LIMB[4] ** 2)))

    def wall_cap(self, ball, ef8: np.ndarray, floor: int, P0: np.ndarray, P: np.ndarray, m: np.ndarray, box,
                 burn: float) -> float:
        """In place: the fire's light on the room (``P0`` unlit, ``P`` lit, linear; the arm's matte ``m`` left out) held
        down round the burning ``ball`` (most at its edge, none from WALL_CAP[3] of its radius past it) so the wall
        within WALL_CAP[0] of its radius past it (over ``box``) is under WALL_CAP[1] of the luma of its limb (its outer
        15 px at 1080), ``burn`` (BURN) of the way to it. Returns the share of its light kept at its edge."""
        _, inner, mid, rf = ball
        y0, y1, x0, x1 = box
        h, w = m.shape
        cx, cy, far = x0 + mid[0], y0 + mid[1], rf * (1.0 + WALL_CAP[3]) + 2.0
        a0, a1 = max(0, int(cy - far)), min(h, int(cy + far) + 1)
        b0, b1 = max(0, int(cx - far)), min(w, int(cx + far) + 1)
        yy, xx = np.mgrid[a0:a1, b0:b1].astype(np.float32)
        near = np.clip(1.0 - (np.hypot(xx - cx, yy - cy) - rf) / (WALL_CAP[3] * rf), 0.0, 1.0) * (1.0 - m[a0:a1, b0:b1])
        u = self.limb(self.exposed(ef8[::2, ::2], floor)[1], inner[::2, ::2], burn)
        inner = inner[::2, ::2]
        rim = (inner >= 0) & (inner < 15.0 * self.px)
        wall = (inner < 0) & (inner > -WALL_CAP[0] * rf) & (m[y0:y1:2, x0:x1:2] < 0.1)
        if rim.sum() < 20 or wall.sum() < 20:
            return 1.0
        luma = np.array([0.299, 0.587, 0.114], np.float32)  # (as a judge's eye: display values, Rec. 601)
        edge = float((ramp_lin(u[rim]) ** np.float32(1 / 2.2) @ luma).mean())
        lo, hi = ((a[y0:y1:2, x0:x1:2][wall] ** np.float32(1 / 2.2) @ luma).mean() for a in (P0, P))
        most = WALL_CAP[1] * edge
        if hi <= most or hi <= lo:
            return 1.0
        ys, xs = np.nonzero(wall)
        at = near[np.clip(y0 + 2 * ys - a0, 0, a1 - a0 - 1), np.clip(x0 + 2 * xs - b0, 0, b1 - b0 - 1)]
        share = (1.0 - (most - lo) / (hi - lo)) / max(float(at.mean()), 0.1)  # (of its light held at its edge)
        kept = 1.0 - burn * (1.0 - float(np.clip(1.0 - share, WALL_CAP[2], 1.0)))
        lit = P[a0:a1, b0:b1]
        lit -= P0[a0:a1, b0:b1]
        lit *= (1.0 - (1.0 - kept) * near)[..., None]
        lit += P0[a0:a1, b0:b1]
        return kept

    def layer(self, Q: np.ndarray, R: np.ndarray, ef8: np.ndarray, wf: np.ndarray, box, floor: int, burn: float,
              body: float, since: float, ball, pulse: float = 1.0) -> None:
        """In place over ``box``: the composite ``Q`` (linear) with the render's fire drawn on it as a light (``ef8``:
        the render's box as placed, before its soot was made dense; ``wf``: its fire weight), over ``R``, the lit room
        (linear), ``burn`` (BURN) of the way to the burning ``ball`` (Fire.ball, or None) and ``body`` (BODY) of the way
        to it as one dying body."""
        y0, y1, x0, x1 = box
        q, room = Q[y0:y1, x0:x1], R[y0:y1, x0:x1]
        v, u = self.exposed(ef8, floor)
        u += np.float32(pulse ** (1 / 2.2) - 1.0) * u * (1.0 - u)
        # A retained fire key must not turn cooled neutral smoke into a flat brown emissive shape. Chroma keeps
        # dark orange folds; source brightness keeps the white-hot core even where it is almost neutral.
        rgb = np.maximum(ef8.astype(np.float32) - floor, 0.0)
        warmth = (rgb[..., 0] - rgb[..., 1]) / np.maximum(rgb.sum(-1), 1.0)
        hot_material = np.maximum(C._smoothstep(0.04, 0.16, warmth), C._smoothstep(0.55, 0.9, v))
        W = wf
        bands = (1.0 - burn) * (1.0 - C._smoothstep(*SOOT_BANDS[3], since))
        if bands > 0.01:  # the soot takes the render's own billows (where it darkens the room, not the fire)
            q = q * self.billows(v, q, room, W, bands)[..., None]
        dying = 1.0 - burn
        if dying > 0:
            # Near live fire, fine soot scatters its orange light. Preserve its billow luminance while warming
            # the neutral skirt; otherwise the last fire floats in a flat grey fringe with pale cream pockets.
            lq, lr = q @ C._LUMA, room @ C._LUMA
            heat = np.clip(4.0 * wide_blur(W * (ramp_lin(u) @ C._LUMA), 0.035 * self.side), 0.0, 1.0)
            soot = np.clip((lr - lq) / np.maximum(lr, 1e-4), 0.0, 1.0) * (1.0 - W)
            hue = np.array([1.0, 0.32, 0.08], np.float32)
            q += (np.float32(0.65 * dying) * heat * soot)[..., None] * (lq[..., None] * (hue / (hue @ C._LUMA)) - q)
        inside, fire, own = np.zeros_like(W), W, np.zeros_like(W)
        # The broad ball stays hot while its illumination is already falling. Keep the phone treatment through
        # that phase; releasing at half burn abruptly restored crisp texture and a filled cooling contour.
        fringe = float(C._smoothstep(3.0, 7.0, since) * C._smoothstep(0.05, 0.35, burn))
        if ball is not None:
            A, inner, mid, radius = ball
            glare_alpha = A
            margin_depth = min(FLAME_EDGE[0] * self.px, 0.27 * radius)
            if fringe > 0:
                A = self.fringe(A, inner, mid, radius, since, fringe)
            if burn > 0:
                u = self.limb(u, inner, burn, A)
                q = q + (burn * (1.0 - C._smoothstep(0.9, 1.0, A)))[..., None] * (room - q)  # (round it while it
                # burns, out to its edge, the wall it lights: the composite's dull ball left in its feather drew a dark
                # hairline)
            # The last flame does not outline the entire smoke volume. Thin its limb by the source's own hot
            # folds, dissolving dim stretches into soot continuously as the fire cools.
            edge = C._smoothstep(0.0, 24.0 * self.px, inner)
            A = A * (burn + dying * (edge + (1.0 - edge) * C._smoothstep(0.35, 0.75, u)))
            A *= burn + dying * hot_material
            if dying > 0:
                # Dark fire folds belong over actual soot. The held ball must not fill the outline's cleared
                # wisps back in as opaque brown lobes; genuinely bright fire can still shine through them below.
                density = np.clip((lr - lq) / np.maximum(lr, 1e-4), 0.0, 1.0)
                A *= burn + dying * C._smoothstep(0.05, 0.4, density)
            own = body * A
            inside = (burn + dying * body) * A
            fire = np.maximum(inside, np.float32(dying) * W * C._smoothstep(0.35, 0.7, u))  # only hot pockets outside its body
        fire = fire * (burn + dying * hot_material)
        # as it dies, its smoke and the dim edges of its fire taken down the fire's colours (red-brown folds), over
        # its soot: the ball's own brown-black, or the soot round it (its embers glow in it, never ringed darker)
        F = ramp_lin(u if dying <= 0 else u * (1.0 - np.float32(dying * (1.0 - BODY_SOOT[2])) * (1.0 - W)))
        source_F = F
        if ball is not None and fringe > 0:
            # Normalize over the original flame support so unresolved dark detail softens without drawing wall
            # colour into the core. Keep the unfiltered source for lens scatter and all scene illumination.
            def resolved(sigma):
                blur = lambda a: cv2.GaussianBlur(a, (0, 0), max(0.25, sigma * self.px))
                return blur(u * glare_alpha) / np.maximum(blur(glare_alpha), 1e-4)

            fine, broad = (resolved(sigma) for sigma in FIRE_RESOLVE[:2])
            phone_u = fine + np.float32(FIRE_RESOLVE[2]) * (broad - fine)
            phone_u *= 1.0 - np.float32(dying * (1.0 - BODY_SOOT[2])) * (1.0 - W)
            F = F + np.float32(fringe) * (ramp_lin(phone_u) - F)
        emit = fire[..., None] * F
        if ball is not None and fringe > 0:
            # Blur opacity and emitted light together so the lens cannot draw a second coloured contour.
            margin = np.float32(fringe) * (1.0 - C._smoothstep(0.0, margin_depth, inner))
            sigma = max(0.25, FLAME_EDGE[2] * self.px)
            fire += margin * (cv2.GaussianBlur(fire, (0, 0), sigma) - fire)
            emit += margin[..., None] * (cv2.GaussianBlur(emit, (0, 0), sigma) - emit)
        out = q * (1.0 - fire[..., None]) + emit
        at = _around(fire > 0.01, int(0.1 * self.side)) if dying > 0 else None  # (only round the fire)
        if at is not None:
            under = self.soot(q[at], room[at], W[at], own[at])
            out[at] += (np.float32(dying) * fire[at])[..., None] * under
        if ball is not None and fringe > 0:
            # Opaque fire also replaces the plate's sensor texture. Restore the lost noise variance across its
            # core and tongues in linear light; the median excludes room edges rather than copying them.
            display = np.maximum(room, 0.0) ** np.float32(1 / 2.2)
            residual = display - cv2.GaussianBlur(display, (0, 0), max(0.5, 0.7 * self.px))
            grain = float(np.clip(1.4826 * np.median(np.abs(residual)), 0.35 / 255, 1.5 / 255))
            noise = np.random.default_rng([self.seed, 193, int(since)]).standard_normal(fire.shape).astype(np.float32)
            weight = fringe * np.sqrt(np.maximum(fire * (2.0 - fire), 0.0))
            out += (2.2 * grain * weight * noise)[..., None] * np.maximum(out, 1e-5) ** np.float32(1 - 1 / 2.2)
            # Retain the source's lens scatter while its directly resolved margin becomes translucent.
            glare = np.maximum(fire, np.float32(fringe) * glare_alpha)[..., None] * source_F
        else:
            glare = emit
        Q[y0:y1, x0:x1] = out + self.glow(glare, burn, inside)

    def soot(self, q: np.ndarray, room: np.ndarray, W: np.ndarray, body: np.ndarray) -> np.ndarray:
        """(linear, the size of its inputs) what is under the dying fire (``W``: its weight): in its ``body`` (in
        [0, 1]) the ball's own soot (BODY_SOOT: brown-black, a share of the lit room's luminance round it); outside it
        the composite's soot round it (``q`` where it darkens the ``room`` and is neither fire nor the ball), or where
        there is none near, the composite itself (a wisp of fire over the bare wall is its light, not a dark strand)."""
        hue = (np.array(BODY_SOOT[1], np.float32) / 255.0) ** np.float32(2.2)
        lr = room @ C._LUMA
        own = (np.float32(BODY_SOOT[0]) * wide_blur(lr, 0.03 * self.side))[..., None] * (hue / (hue @ C._LUMA))
        near = (1.0 - W) * (1.0 - body) * np.clip(2.0 * (lr - q @ C._LUMA) / np.maximum(lr, 1e-4), 0.0, 1.0)
        sigma = 0.02 * self.side
        much = wide_blur(near, sigma)
        round_it = wide_blur(q * near[..., None], sigma) / np.maximum(much, 1e-3)[..., None]
        outside = q + C._smoothstep(0.03, 0.2, much)[..., None] * (round_it - q)
        return outside + body[..., None] * (own - outside)

    def glow(self, emit: np.ndarray, burn: float, inside: np.ndarray) -> np.ndarray:
        """(box size, linear) the light of the fire drawn (``emit``) scattered round it: its highlights spread by the
        lens (BLOOM) and, as it dies, its embers' glow in the soot round them (EMBER_GLOW, in one fire colour, less of
        it from the ball, ``inside``, and over it, so its folds stay dark)."""
        hb, wb = emit.shape[:2]
        y = emit @ C._LUMA
        out = np.zeros_like(emit)
        top = float(y.max())
        if top > BLOOM[0]:
            src = emit * (np.maximum(y - np.float32(BLOOM[0]), 0.0) / np.maximum(y, 1e-4))[..., None]
            for (sigma, gain), k in zip(BLOOM[1], (2, 4)):  # (at a half and a quarter size)
                small = cv2.resize(src, (max(2, wb // k), max(2, hb // k)), interpolation=cv2.INTER_AREA)
                out += np.float32(gain) * cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma * self.px / k), (wb, hb),
                                                     interpolation=cv2.INTER_LINEAR)
        if burn < 1.0 and top > 1e-4:
            less = 1.0 - np.float32(1.0 - EMBER_GLOW[3]) * inside
            small = cv2.resize(y * less, (max(2, wb // 4), max(2, hb // 4)), interpolation=cv2.INTER_AREA)
            spread = sum(np.float32(share) * cv2.GaussianBlur(small, (0, 0), sigma * self.px / 4)
                         for sigma, share in EMBER_GLOW[0])
            hue = np.array(EMBER_GLOW[2], np.float32)
            spread = cv2.resize(spread, (wb, hb), interpolation=cv2.INTER_LINEAR) * less
            out += (np.float32((1.0 - burn) * EMBER_GLOW[1]) * spread)[..., None] * (hue / (hue @ C._LUMA))
        return out

    def billows(self, v: np.ndarray, q: np.ndarray, room: np.ndarray, W: np.ndarray, weight: float) -> np.ndarray:
        """(box size) the gain on the composite ``q`` (linear) that puts the render's own billows (``v``: its display
        luma) on its soot: a band-pass of its smoke's log luma (SOOT_BANDS) as display levels, as much as the soot
        darkens the ``room`` and is not fire (``W``), times ``weight``."""
        hb, wb = q.shape[:2]
        h2, w2 = max(2, hb // 2), max(2, wb // 2)
        y = np.log(np.maximum(cv2.resize(v, (w2, h2), interpolation=cv2.INTER_AREA), 2.0 / 255))
        sm = cv2.resize(1.0 - W, (w2, h2), interpolation=cv2.INTER_AREA)
        band = smoke_blur(y, sm, SOOT_BANDS[0] * self.px / 2) - smoke_blur(y, sm, SOOT_BANDS[1] * self.px / 2)
        lq, lr = q @ C._LUMA, room @ C._LUMA
        soot = np.clip(2.0 * (lr - lq) / np.maximum(lr, 1e-4), 0.0, 1.0) * (1.0 - W)
        at = cv2.resize(soot, (w2, h2), interpolation=cv2.INTER_AREA) > 0.5
        spread = float(band[at].std()) if at.sum() > 50 else 0.0
        if spread < 1e-4:
            return np.ones((hb, wb), np.float32)
        lift = cv2.resize(np.clip(band / spread, -2.5, 2.5), (wb, hb), interpolation=cv2.INTER_LINEAR) \
            * np.float32(SOOT_BANDS[2] / 255 * weight) * soot
        shown = np.maximum(lq, 1e-5) ** np.float32(1 / 2.2)
        return (np.clip(1.0 + lift / np.maximum(shown, 0.03), 0.5, 2.0) ** np.float32(2.2)).astype(np.float32)


_GAM8 = (((np.arange(2 * _STEPS + 1, dtype=np.float64) / _STEPS) ** 2).clip(0.0, 1.0) ** (1 / 2.2) * 255.0
         + 0.5).astype(np.uint8)  # 8-bit display values by the square root of linear light (_by_root)


def shoulder_added(P: np.ndarray, plate8: np.ndarray) -> np.ndarray:
    """``P`` (linear) as 8-bit display values, what the blast added to the room (``plate8``, the plate) rolled off
    over the shot's highlight shoulder (bending.SHOULDER, its tanh), each channel; what it darkened and the room's own
    highlights as they were (rolled off again, the phone's highlights, the skin's red among them, dimmed for as long
    as the blast lasted)."""
    curve, _, _, curve8 = _tables()
    room = lin(plate8)
    return _by_root(_GAM8, np.where(P > room, room + _by_root(curve, P) - curve8[plate8], P))


def library_element() -> np.ndarray:
    """The explosion rendered ahead (the made-ahead library's explosion/element.mp4)."""
    from .bending import library_dir
    from .media import read_video

    lib = library_dir()
    path = None if lib is None else Path(lib) / "explosion" / "element.mp4"
    if path is None or not path.exists():
        raise FileNotFoundError("no explosion in the made-ahead library (samples/library/bending/explosion)")
    return read_video(path).frames


# ------------------------------------------------------------------ the shot
def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline, form: str = "fist", seed: int = 0,
           progress=None, element: np.ndarray | None = None) -> np.ndarray:
    """The explosion on the plate (uint8 RGB [T, H, W, 3]), set off on the loudest hit (as big as it is loud),
    shaken and punched by its blast. ``element``: the fireball rendered over black (default: the library's)."""
    say = progress or (lambda stage, frac, msg: None)
    n, h, w = plate.shape[:3]
    side = min(h, w)
    if not timeline.hits:
        return plate.copy()
    hero = int(np.argmax(timeline.sizes))
    k0, size = int(np.clip(timeline.hits[hero], 0, n - 1)), float(timeline.sizes[hero])
    origin, aim, s = blast_point(track, k0, "palm" if form == "palm" else "fist", timeline.aim)
    origin = inside(origin, w, h)
    cam = timeline.camera
    el = library_element() if element is None else element
    floor = C.estimate_black_level(el)
    from .bending import border_fade

    # the render's own frame edges fade out (its smoke rising past the top of the render drew a straight cut across
    # the room once carried there)
    edge = border_fade(el.shape[1], el.shape[2], EDGE_FADE)[..., None]
    faded = np.empty_like(el)
    for e in range(len(el)):
        faded[e] = (floor + (el[e].astype(np.float32) - floor) * edge).clip(0, 255).astype(np.uint8)
    el = faded
    e_ign, el_centre, el_r8 = render_blast(el, floor)
    R = min(SIZE * s * size, MOST * side)
    k_el = min(R / el_r8, CLOUD_MOST * side / max(render_reach(el, floor, el_centre, e_ign) * (1 + SWELL[0]), 1.0))
    R = k_el * el_r8
    smoke_ref = C.smoke_reference_luma(el, PARAMS, floor, True)
    shading_ref = C.smoke_shading_ref(el, PARAMS, floor, True, smoke_ref)
    billow_ref = soot_shading_ref(el, floor, e_ign + 3 * GROW_FRAMES)
    env_rgb = C.environment_colour(plate[0], el, floor, PARAMS)
    amb = C.ambient_map(plate[k0], PARAMS)
    volume_amb = amb
    if mask is not None:
        # The nearby wall illuminates airborne soot. Blurring the foreground hand into the ambient map left a
        # stationary pink underside after the fire went out; exclude it with a normalized blur before lighting soot.
        small_size = (max(2, w // 8), max(2, h // 8))
        uncovered = cv2.resize(1.0 - hand_coverage(plate[k0], mask, k0), small_size, interpolation=cv2.INTER_AREA)
        room_small = lin(cv2.resize(plate[k0], small_size, interpolation=cv2.INTER_AREA))
        sigma = max(1.0, PARAMS.ambient_blur_frac * w / 8)
        weights = cv2.GaussianBlur(uncovered, (0, 0), sigma)
        clean = cv2.GaussianBlur(room_small * uncovered[..., None], (0, 0), sigma) / np.maximum(weights, 0.01)[..., None]
        volume_amb = cv2.resize(clean, (w, h), interpolation=cv2.INTER_LINEAR)
    plate_key = float((plate[k0, ::8, ::8].astype(np.float32) / 255.0 @ C._LUMA).mean())
    some = np.stack([lin(plate[i]) for i in np.linspace(0, n - 1, 8).astype(int)])
    white, tint = fx.plate_white(some), fx.plate_tint(some)
    del some
    rng = np.random.default_rng(seed)
    dust = 0.35 + 0.65 * np.clip(cv2.resize(rng.random((h // 20, w // 20)).astype(np.float32), (w, h),
                                            interpolation=cv2.INTER_CUBIC), 0, 1)
    parts = Particles(origin, aim, s, R, side, seed, count=0.4 + 0.6 * size)
    reach = 1.05 * max(math.hypot(origin[0] - x, origin[1] - y) for x in (0, w) for y in (0, h))

    def frame_of(since: int) -> int:  # the render's growth played BURST_SPEED times faster, then on in real time
        g = GROW_FRAMES / BURST_SPEED
        return e_ign + (int(round(since * BURST_SPEED)) if since <= g else int(round(GROW_FRAMES + (since - g))))

    def lift(t: int) -> tuple[float, float]:
        """(frames since its burst, px its middle has risen by then): it rolls up as it burns, its smoke climbs on."""
        since = max(0.0, (t - k0) - GROW_FRAMES / BURST_SPEED)  # (after its burst, as the render plays in real time)
        climb = since - ROLL[3] * (1.0 - math.exp(-since / ROLL[3]))
        return since, R * (ROLL[0] * (1.0 - math.exp(-since / ROLL[1])) + ROLL[2] * climb / fx.FPS)

    xs_all = np.arange(w, dtype=np.float32)
    columns = cv2.GaussianBlur(rng.standard_normal((1, el.shape[2])).astype(np.float32), (0, 0),
                              0.045 * el.shape[2])[0]
    columns = np.clip(columns / max(float(columns.std()), 1e-6), -2.0, 2.0)
    shapes = SootShapes(el, floor, e_ign, k_el, side)
    burns, bodies = fire_burns(el, floor)

    def element_on(t: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The render on frame ``t`` placed in the room, and its cloud's filled silhouette (SootShapes) with it."""
        e = frame_of(t - k0)
        if t < k0 or e >= len(el):
            return None
        since, up = lift(t)
        grow = 1.0 - math.exp(-since / SWELL[1])
        k = k_el * (1.0 + SWELL[0] * grow)  # the hot cloud swells as it burns out, and rises
        o = (origin[0], origin[1] - up)
        S = np.array([[k, 0, o[0] - k * el_centre[0]], [0, k, o[1] - k * el_centre[1]], [0, 0, 1]])
        Ai = cv2.invertAffineTransform((_between(cam, k0, t) @ S)[:2])
        mid = float(room_point(cam, k0, t, o)[1])
        v = np.arange(h, dtype=np.float32) - mid  # its rows over its middle are stretched up, the rest kept
        rows = mid + v / (1.0 + TOP_STRETCH * grow / (1.0 + np.exp(np.clip(v / (0.25 * R), -50, 50))))
        mx = (Ai[0, 0] * xs_all[None] + Ai[0, 1] * rows[:, None] + Ai[0, 2]).astype(np.float32)
        my = (Ai[1, 0] * xs_all[None] + Ai[1, 1] * rows[:, None] + Ai[1, 2]).astype(np.float32)
        # Unequal buoyancy carries whole columns of the source and its matte together. It breaks the rendered
        # cloud's horizontal lower shelf into rising lobes without sliding texture across a stationary outline.
        loft = C._smoothstep(12.0, 40.0, t - k0) * (1.0 - float(burns[e]))
        if loft > 0:
            drift = np.interp(mx.ravel() + 0.15 * since, np.arange(len(columns)), columns).reshape(mx.shape)
            my += (0.15 * el_r8 * loft * drift).astype(np.float32)
        sil = cv2.remap(shapes.at(e), (mx - 1.5) / 4, (my - 1.5) / 4, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # (from a quarter of the render's size)
        return cv2.remap(el[e], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), sil

    out = plate.copy()
    last = min(n, k0 + max(SHOCK_FRAMES, int(np.max(parts.t0 + parts.life)) + 1,
                           int(math.ceil(GROW_FRAMES / BURST_SPEED + len(el) - e_ign - GROW_FRAMES)) + 1))
    q = 8
    spread, spread_norm = falloff((h // q, w // q), LIGHT_FALLOFF[0] * R / q, R / q, LIGHT_FALLOFF[1])
    flash_spread, flash_norm = falloff((h // q, w // q), LIGHT_FALLOFF[0] * R / q, R / q, LIGHT_SETTLE[2])
    outline, fire = Outline(h, w, side, seed), Fire(side, seed)
    volume = SmokeVolume(h, w, side, seed)
    wobble = np.random.default_rng([seed, 89]).standard_normal((1, n + 8))
    wobble = cv2.GaussianBlur(wobble, (0, 0), FLICKER[1])[0, 4:-4]
    flicker = 1.0 + FLICKER[0] * np.clip(wobble / max(float(wobble.std()), 1e-6), -1.5, 1.5)
    held = held_colour = src = None  # (the light on the frame before, its colour, and where its fire's middle was)

    def prepare(t):
        nonlocal held, held_colour, src
        since = t - k0
        settled = float(C._smoothstep(*LIGHT_SETTLE[:2], since))
        o_t = room_point(cam, k0, t, origin)
        placed = element_on(t)
        E8 = box = ef8 = wf = sil_box = detail8 = smoke = None
        burn, body = (0.0, 0.0) if frame_of(since) >= len(el) else (float(burns[frame_of(since)]),
                                                                    float(bodies[frame_of(since)]))
        if placed is not None:
            E8, sil = placed
            E8 = radial_blur(E8, o_t, GROW_BLUR.get(since, 0.0))
            E8 = cv2.GaussianBlur(E8, (0, 0), DEFOCUS_PX * side / 1080)
            ys_, xs_ = np.nonzero(C._peak(E8[::4, ::4]) > floor + 2)
            if ys_.size:
                margin = max(24, int(math.ceil(0.18 * lift(t)[1] + 4)))
                box = (max(0, 4 * ys_.min() - margin), min(h, 4 * ys_.max() + margin), max(0, 4 * xs_.min() - margin),
                       min(w, 4 * xs_.max() + margin))
                y0, y1, x0, x1 = box
                ef8 = E8[y0:y1, x0:x1].copy()  # (its fire, before its soot is made dense)
                wf = np.clip(C.fire_weight_map(C._black_floor_f(ef8, floor), PARAMS, w, True), 0.0, 1.0)
                cloud = cv2.moments(sil[y0:y1, x0:x1])
                centre = ((x0 + cloud["m10"] / cloud["m00"], y0 + cloud["m01"] / cloud["m00"])
                          if cloud["m00"] > 1e-4 else ((x0 + x1) / 2, (y0 + y1) / 2))
                # One continuous key drives both soot preparation and fire drawing. Holding only the final
                # fire layer still let saturation reopen pale holes in its underlying soot in a single frame.
                wf = fire.material(wf, box, centre, burn, sil[y0:y1, x0:x1])
                sil_box = sil[y0:y1, x0:x1].copy()
                smoke = 1.0 - wf
                E8 = soot_detail(E8, floor, box, smoke)
                volume_mix = float(C._smoothstep(*VOLUME_HANDOFF, since)) * (1.0 - burn)
                detail8 = E8 if volume_mix < 1.0 else None
                E8 = fill_soot(E8, floor, box, sil_box, smoke)
                if volume_mix > 0.0 and LOBE_SEPARATE[0] < since < sum(CLEAR):
                    volume._prepare_noise()
        if box is None:
            fire.previous_material = None
        # the blast's light on the room and the arm, from its own image: the render's fire, and the white-hot core's
        # light on its first frames, spread round it and tinted to its colour
        fire_small, colour = None, fx.kelvin_rgb(4500.0)
        if E8 is not None:
            e = C._black_floor_f(cv2.resize(E8, (w // q, h // q), interpolation=cv2.INTER_AREA), floor)
            fire_small = C._lin(e) * C.fire_weight_map(e, PARAMS, w // q, True)[..., None]
            colour = light_colour(fire_small)
        core_l = CORE_LIGHT.get(since, 0.0) * size
        if core_l > 0:
            rc = (0.35 + 0.3 * since) * s
            yy, xx = np.mgrid[:h // q, :w // q].astype(np.float32) * q
            core_s = (3.0 * core_l * np.exp(-((xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2) / (2 * rc * rc)))[..., None] \
                * fx.kelvin_rgb(4500.0)
            f_y = 0.0 if fire_small is None else float(fire_small.sum())
            c_y = float(core_s.sum())
            colour = (colour * f_y + fx.kelvin_rgb(4500.0) * c_y) / (f_y + c_y)
            fire_small = core_s if fire_small is None else fire_small + core_s
        lit = None
        if fire_small is not None and fire_small.max() > 0:
            y_small = fire_small @ C._LUMA
            if settled <= 0.0:
                lit = blast_light(y_small, flash_spread, flash_norm)
            elif settled >= 1.0:
                lit = blast_light(y_small, spread, spread_norm)
            else:
                lit = blast_light(y_small, (1.0 - settled) * flash_spread / flash_norm
                                  + settled * spread / spread_norm, 1.0)
            ys_, xs_ = np.indices(y_small.shape)
            src = (q * float((xs_ * y_small).sum() / y_small.sum()), q * float((ys_ * y_small).sum() / y_small.sum()))
            held_colour = colour
        if held is not None:
            # Release the superseded flash footprint as the compact source takes over. Its much stronger
            # history otherwise kept even a tightly falling live fire on a broad, saturated orange pedestal.
            hold = LIGHT_HOLD * (1.0 - settled) if since <= LIGHT_SETTLE[1] else LIGHT_HOLD
            lit = hold * held if lit is None else np.maximum(lit, hold * held)
        if lit is not None and lit.max() > 1e-4:
            held, colour = lit, held_colour
        else:
            held = lit = None  # a released or exhausted flash must not return when ordinary light smoothing resumes
        return t, E8, box, ef8, wf, sil_box, detail8, smoke, lit, colour, src, burn, body

    def draw(t, E8, box, ef8, wf, sil_box, detail8, smoke, lit, colour, src, burn, body):
        since = t - k0
        settled = float(C._smoothstep(*LIGHT_SETTLE[:2], since))
        pulse = 1.0 + (flicker[t] - 1.0) * min(1.0, since / 3.0)
        P = lin(plate[t])
        m = hand_coverage(plate[t], mask, t)
        guard = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 2.0)
        o_t = room_point(cam, k0, t, origin)
        amb_t, billow, rim_maps, ball = amb, None, None, None
        if box is not None:
            y0, y1, x0, x1 = box
            if detail8 is not None:
                billow = soot_billows(detail8, floor, box, billow_ref, smoke)
                r_in = max(3, int(BILLOW_IN * side) | 1)
                inner = cv2.erode(sil_box, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_in, r_in)))
                billow = billow * cv2.GaussianBlur(inner, (0, 0), r_in / 2)
            rim_maps = outline.maps(E8, floor, box, since, lift(t)[1], fire_weight=wf)
            ball = fire.ball(wf, burn, C._peak(ef8) > floor + 16) if burn > 0 or body > 0 else None
        if lit is not None:
            L = cv2.resize(lit, (w, h)) * np.float32(LIGHT_GAIN)
            rim = rim_light(m, src) if mask is not None else np.zeros((h, w), np.float32)  # (from the fire's middle)
            on = (1.0 - m) + m * (ARM_LIGHT[0] + (ARM_LIGHT[1] + ARM_LIGHT[2] * burn) * rim)
            shone = P + (fx.albedo(P, white, tint) * (L * on)[..., None] + np.float32(LIGHT_GLOW) * L[..., None]) \
                * colour
            hot = shone > LIGHT_KNEE  # (the composite takes the plate in 8 bits: past white it would clip flat)
            shone[hot] = LIGHT_KNEE + (1.0 - LIGHT_KNEE) * np.tanh((shone[hot] - LIGHT_KNEE) / (1.0 - LIGHT_KNEE))
            P0, P = P, warm_light(P, shone, m, rim, burn, response=pulse, relief=settled)
            if ball is not None and burn > 0:  # (the wall by the burning ball lit no brighter than the ball's limb)
                fire.wall_cap(ball, ef8, floor, P0, P, m, box, burn)
            amb_t = amb + np.float32(SOOT_LIT) * L[..., None] * colour
        # Keep the illuminated current plate for one foreground restoration after the rear heat and pressure
        # distortion. Restoring first let those remaps pull fire and soot over the fingers again.
        hand = P
        # the fireball and its smoke, behind the arm
        if E8 is not None:
            volume_mix = float(C._smoothstep(*VOLUME_HANDOFF, since)) * (1.0 - burn)
            # Once the volume owns the smoke, the older composite would be built only to be overwritten. Keep
            # the illuminated plate outside it and spend that CPU time resolving depth inside the cloud instead.
            if volume_mix < 1.0:
                comp = C.composite_frame(_by_root(_GAM8, P), E8, PARAMS, holdout_row=None, emission=tuple(o_t),
                                         fire_energy_ref=None, plate_key=plate_key, grain_sigma=1.5, black_floor=floor,
                                         fire_effect=True, dense_smoke=True, smoke_ref=smoke_ref, env_rgb=env_rgb,
                                         seed=seed + t, ambient=amb_t, shading_ref=shading_ref, smoke_albedo=SOOT_ALBEDO,
                                         g_smoke=1.0 - _ease((since - CLEAR[0]) / CLEAR[1]), thin_soot=True)
                Q = lin(comp)
                if detail8 is not None:  # (as the smoke clears, its billows go with it)
                    y0, y1, x0, x1 = box
                    lit = Q[y0:y1, x0:x1] * (1.0 + (1.0 - _ease((since - CLEAR[0]) / CLEAR[1])) * billow)[..., None]
                    Q[y0:y1, x0:x1] = np.minimum(lit, np.maximum(Q[y0:y1, x0:x1], P[y0:y1, x0:x1]))
                if rim_maps is not None:
                    outline.finish(Q, P, box, rim_maps)
            else:
                Q = P.copy()
            if box is not None:  # its soot scattering the room's light; its fire as a light (the composite drew it as
                # lit smoke)
                if volume_mix < 1.0:
                    scatter(Q, P, box, side)
                if volume_mix > 0.0:
                    clear = 1.0 - _ease((since - CLEAR[0]) / CLEAR[1])
                    erosion = None
                    if rim_maps is not None:
                        _, wisp, eat, thin, _, _, _ = rim_maps
                        erosion = cv2.resize(C._smoothstep(eat - 0.22, eat + 0.22, wisp) * thin,
                                             (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                    v = volume.draw(P[y0:y1, x0:x1], ef8, sil_box, floor,
                                    (volume_amb[y0:y1, x0:x1] + amb_t[y0:y1, x0:x1] - amb[y0:y1, x0:x1]), since, lift(t)[1], clear, box, erosion)
                    Q[y0:y1, x0:x1] += np.float32(volume_mix) * (v - Q[y0:y1, x0:x1])
                fire.layer(Q, P, ef8, wf, box, floor, burn, body, since, ball, pulse=pulse)
            P = Q
            if since >= 1:  # the hot air round and above the fireball shimmers
                hot = np.clip(fx._lum(lin(cv2.resize(E8, (w // 4, h // 4), interpolation=cv2.INTER_AREA))) * 4.0, 0, 1)
                hazed = fx.heat_haze(P[None], cv2.resize(hot, (w, h))[None], seed=seed, frame0=t)[0]
                P += (1.0 - guard)[..., None] * (hazed - P)
        core_i = CORE.get(since, 0.0) * size
        if core_i > 0:  # the white-hot core and its glow behind the arm; the lens's veil over everything
            rc = (0.35 + 0.3 * since) * s
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            r2 = (xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2
            core = (2.5 * core_i * np.exp(-r2 / (2 * rc * rc)))[..., None] * fx.kelvin_rgb(5200.0)
            glow = (0.18 * core_i * np.exp(-r2 / (2 * (3.0 * rc) ** 2)))[..., None] * fx.kelvin_rgb(3200.0)
            P = P + core * (1 - m[..., None]) + glow * (1 - 0.6 * m[..., None]) + np.float32(0.015 * core_i) \
                * fx.kelvin_rgb(3000.0)
        if since < SHOCK_FRAMES:
            shocked = shock_ring(P, o_t, aim, since, reach * (0.5 + 0.5 * size), side, seed,
                                 fx.kelvin_rgb(2600.0) * 0.6, dust)
            P += (1.0 - guard)[..., None] * (shocked - P)
        arm = cv2.boundingRect((m > 0).astype(np.uint8))
        if arm[2] and arm[3]:
            a = (slice(arm[1], arm[1] + arm[3]), slice(arm[0], arm[0] + arm[2]))
            P[a] += m[a][..., None] * (np.maximum(hand[a], lin(plate[t][a])) - P[a])
        M = _between(cam, k0, t)
        for front in (False, True):  # behind the arm and inside the fireball's body, or flying towards the camera
            drawn = parts.draw(since, (h, w), M, front)
            if drawn is not None:
                emit, (a0, a1, b0, b1) = drawn
                if not front:
                    cover = 0.0 if E8 is None else \
                        np.clip((C._peak(E8[a0:a1, b0:b1]).astype(np.float32) / 255.0 - 0.06) / 0.3, 0, 1) * 0.9
                    emit = emit * ((1 - m[a0:a1, b0:b1]) * (1 - cover))[..., None]
                P[a0:a1, b0:b1] += emit
        stops = FLASH.get(since, 0.0) * size
        if stops > 0:  # the blast's flash lights what is near it most (a uniform wash greyed the hand to a ghost)
            yy, xx = np.mgrid[:h:4, :w:4].astype(np.float32)
            near = np.exp(-((xx - o_t[0]) ** 2 + (yy - o_t[1]) ** 2) / (2 * (FLASH_REACH * max(R, s)) ** 2))
            k = 1.0 + (2.0 ** stops - 1.0) * (FLASH_FAR + (1.0 - FLASH_FAR) * near)
            P = P * cv2.resize(k.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        out[t] = shoulder_added(P, plate[t])
        return t

    def finished(t):
        say("compositing", 0.7 + 0.14 * (t - k0 + 1) / max(1, last - k0), f"The explosion: frame {t + 1} of {last}")

    # Only the shape, material and light histories are sequential. Bound queued frame scratch to the pool's size.
    if fx.THREADS == 1:
        for t in range(k0, last):
            finished(draw(*prepare(t)))
    else:
        with ThreadPoolExecutor(fx.THREADS) as ex:
            pending = deque()
            for t in range(k0, last):
                pending.append(ex.submit(draw, *prepare(t)))
                if len(pending) >= fx.THREADS:
                    finished(pending.popleft().result())
            for task in pending:
                finished(task.result())
    out = fx.shake(out, [k0], [size], px=KNOCK_PX * h / 1080, seed=seed)  # the camera knocked...
    for t in range(k0, min(n, k0 + 8)):  # ...and pushed by the pressure wave
        z = ZOOM * size * math.exp(-(((t - k0) - 2) / 1.6) ** 2)
        if z > 1e-4:
            o_t = room_point(cam, k0, t, origin)
            Mz = cv2.getRotationMatrix2D((float(o_t[0]), float(o_t[1])), 0.0, 1.0 + z)
            out[t] = cv2.warpAffine(out[t], Mz, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _around(mask: np.ndarray, margin: int):
    """The rows and columns (slices) of the box round ``mask``'s True pixels grown by ``margin`` px, or None."""
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    if rows.size == 0:
        return None
    h, w = mask.shape
    return (slice(max(0, int(rows[0]) - margin), min(h, int(rows[-1]) + margin + 1)),
            slice(max(0, int(cols[0]) - margin), min(w, int(cols[-1]) + margin + 1)))


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _between(cam, a: int, b: int) -> np.ndarray:
    from .bending import camera_between

    return camera_between(cam, a, b)
