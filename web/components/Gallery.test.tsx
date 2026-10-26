import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Gallery from "./Gallery";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
const shot = (id: string, title: string, again?: number) => ({ id, title, again, status: "done", files: { "final.mp4": `/${id}.mp4` } });

beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe("Gallery", () => {
  it("shows a render once: its further performances play together under One render", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => [
      shot("explosion", "BOOOOM"), shot("palm", "BOOM", 0), shot("pfpf", "pf, pf, BOOM", 1), shot("unfa", "Ka-Boom!", 4),
      shot("walk", "fffff"),
    ] }));
    await act(async () => { root.render(<Gallery />); });
    expect([...host.querySelectorAll(".tile video")].map((v) => v.getAttribute("src"))).toEqual(["/explosion.mp4", "/palm.mp4", "/walk.mp4"]);
  });
});
