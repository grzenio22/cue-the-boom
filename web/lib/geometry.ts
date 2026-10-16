/** Map clicks on a letterboxed (object-fit: contain) video to normalised [x, y] on the video content. */

export interface Box { width: number; height: number }
export interface Rect { x: number; y: number; width: number; height: number }

export function contentRect(box: Box, videoW: number, videoH: number): Rect {
  const scale = Math.min(box.width / videoW, box.height / videoH);
  const width = videoW * scale, height = videoH * scale;
  return { x: (box.width - width) / 2, y: (box.height - height) / 2, width, height };
}

const r4 = (v: number) => Math.round(v * 10000) / 10000;

export function normalizedPoint(clickX: number, clickY: number, box: Box, videoW: number, videoH: number): [number, number] | null {
  const r = contentRect(box, videoW, videoH);
  const x = (clickX - r.x) / r.width, y = (clickY - r.y) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return [r4(x), r4(y)];
}
