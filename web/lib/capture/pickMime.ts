/** MediaRecorder container preference: MP4 (Safari, newer Chrome) first, then WebM. */
const CANDIDATES = [
  "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
  "video/mp4",
  "video/webm;codecs=vp9,opus",
  "video/webm",
];

export function pickMime(isTypeSupported: (t: string) => boolean): string | null {
  for (const t of CANDIDATES) {
    try {
      if (isTypeSupported(t)) return t;
    } catch {
      /* some browsers throw on unknown types */
    }
  }
  return null;
}
