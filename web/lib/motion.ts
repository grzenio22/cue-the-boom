/** Whether the viewer asked for less motion (no window, as when rendered on the server: no). */
export function reducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** A tile's clip comes alive while a mouse points at it: it plays muted, looping, and stops where it is when the
 * pointer leaves. Not for touch (a tap opens the tile), nor when the viewer asked for less motion. */
export function peek(video: HTMLVideoElement | null | undefined, on: boolean): void {
  if (!video) return;
  if (!on) {
    if (!video.paused) video.pause();
    return;
  }
  if (reducedMotion()) return;
  video.muted = true;
  video.loop = true;
  const playing = video.play() as Promise<void> | undefined; // (undefined where play() is not implemented)
  playing?.catch(() => undefined);
}

/** The pointer handlers that make a tile's own <video> peek. */
export const peekHandlers = {
  onPointerEnter: (e: { pointerType: string; currentTarget: Element }) => {
    if (e.pointerType === "mouse") peek(e.currentTarget.querySelector("video"), true);
  },
  onPointerLeave: (e: { currentTarget: Element }) => peek(e.currentTarget.querySelector("video"), false),
};
