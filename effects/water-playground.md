# Water

Water is one of the studio's five elements. It is made on your computer from your hand and your voice, with no model generation and no GPU.

Choose **Water**, then a form: **Whip**, **Jet**, **Orb**, **Ribbon**, **Surge** or **Tidal shield**. Hold a **shhh** to form it and add a **sploosh** accent to throw it. The prepared palm supplies the tracking; on your own tracked clip your hand's movements shape it as well: sweep to cast a blade, lift to raise a wave, or drop your hand for a splash crown. Water appears in the finished shot. The live webcam preview supports fire and lightning.

[`water.py`](../cuesync/src/cuesync/water.py) treats water as a lens and a mirror over the plate. Each frame's water is a set of metaballs read as a thickness map, and each pixel's ray is refracted through it to the hand or the room behind, with the room reflected at its rim, the key light's glints, the light it focuses onto the palm inside its shadow, motion blur, and wet skin where it has been. Its colour comes from the room's own light, with a floor on that light and a faint blue-green absorption, so it stays readable in a dim room.

This is a stylised composite from a single camera view, not a physical liquid simulation. Broad sheets, such as the tidal shield, can still look synthetic.

The gallery plays all six forms on the author's dim-room palm with the sample shhhhh · sploosh cue. Retained water comparisons and their source attribution are documented in [the media credits](../samples/CREDITS.md).
