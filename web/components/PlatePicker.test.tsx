import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PlatePicker, { type PickedPlate } from "./PlatePicker";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
let probes: HTMLVideoElement[];
let replies: ((response: Response) => void)[];
let copies: number;
const onPick = vi.fn<(plate: PickedPlate) => void>();
const revoke = vi.fn<(url: string) => void>();
const samples = [{ name: "hand_palm.mp4", url: "/hand.mp4", title: "Your hand: open palm" }];

beforeEach(() => {
  probes = [];
  replies = [];
  copies = 0;
  onPick.mockClear();
  revoke.mockClear();
  vi.stubGlobal("URL", class extends URL {
    static createObjectURL = (blob: Blob | MediaSource) => blob instanceof File ? `blob:${blob.name}` : `blob:copy-${++copies}`;
    static revokeObjectURL = revoke;
  });
  vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => replies.push(resolve))));
  const createElement = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation((name, options) => {
    const element = createElement(name, options);
    if (name === "video") probes.push(element as HTMLVideoElement);
    return element;
  });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function PickAndShow({ unmount }: { unmount: boolean }) {
  const [plate, setPlate] = useState<PickedPlate | null>(null);
  return <>{(!unmount || !plate) && <PlatePicker samples={samples} onPick={(p) => { onPick(p); setPlate(p); }} />}
    <p data-current>{plate?.title}</p></>;
}

const show = async (unmount = false) => { await act(async () => { root.render(<PickAndShow unmount={unmount} />); }); };
const probe = (url: string, kind: "play" | "duration") => probes.find((v) => v.getAttribute("src") === url
  && (kind === "play" ? v.onloadeddata : v.onloadedmetadata));
const pickFile = async (name: string) => {
  const input = host.querySelector<HTMLInputElement>("input[type=file]")!;
  Object.defineProperty(input, "files", { configurable: true, value: [new File(["clip"], name, { type: "video/quicktime" })] });
  await act(async () => { input.dispatchEvent(new Event("change", { bubbles: true })); });
};
const failPlayback = async (name: string) => {
  await act(async () => { probe(`blob:${name}`, "play")!.dispatchEvent(new Event("error")); });
};
const finishDuration = async (url: string) => {
  const video = probe(url, "duration");
  if (!video) return;
  Object.defineProperty(video, "duration", { configurable: true, value: 3 });
  await act(async () => { video.dispatchEvent(new Event("loadedmetadata")); });
};
const finishCopy = async (index: number) => {
  await act(async () => { replies[index]({ ok: true, blob: async () => new Blob(["copy"]) } as Response); });
  await finishDuration(`blob:copy-${copies}`);
};

describe("PlatePicker", () => {
  it.each([false, true])("keeps a sample selected while an older browser copy finishes (unmount: %s)", async (unmount) => {
    await show(unmount);
    await pickFile("first.mov");
    await failPlayback("first.mov");
    expect(host.querySelector("p.chip")?.textContent).toContain("making a copy");
    await act(async () => { host.querySelector<HTMLButtonElement>("button.tile")!.click(); });
    await finishCopy(0);
    expect(onPick.mock.calls.map(([p]) => p.title)).toEqual(["Your hand: open palm"]);
    expect(host.querySelector("[data-current]")?.textContent).toBe("Your hand: open palm");
    expect(revoke).toHaveBeenCalledWith("blob:copy-1");
    expect(host.querySelector("p.chip")).toBeNull();
  });

  it("keeps the newest file when browser copies finish out of order", async () => {
    await show();
    await pickFile("first.mov");
    await failPlayback("first.mov");
    await pickFile("second.mov");
    await failPlayback("second.mov");
    await finishCopy(1);
    await finishCopy(0);
    expect(onPick.mock.calls.map(([p]) => p.title)).toEqual(["second.mov"]);
    expect(host.querySelector("[data-current]")?.textContent).toBe("second.mov");
    expect(revoke).toHaveBeenCalledWith("blob:copy-2");
    expect(revoke).not.toHaveBeenCalledWith("blob:copy-1");
  });

  it("discards a duration probe when a newer file has been selected", async () => {
    await show();
    await pickFile("first.mov");
    const first = probe("blob:first.mov", "play")!;
    Object.defineProperty(first, "videoWidth", { configurable: true, value: 640 });
    await act(async () => { first.dispatchEvent(new Event("loadeddata")); });
    await pickFile("second.mov");
    await failPlayback("second.mov");
    await finishCopy(0);
    await finishDuration("blob:first.mov");
    expect(onPick.mock.calls.map(([p]) => p.title)).toEqual(["second.mov"]);
    expect(revoke).toHaveBeenCalledWith("blob:first.mov");
  });

  it("discards a pending playback check when a sample is picked", async () => {
    await show();
    await pickFile("first.mov");
    await act(async () => { host.querySelector<HTMLButtonElement>("button.tile")!.click(); });
    await failPlayback("first.mov");
    expect(fetch).not.toHaveBeenCalled();
    expect(onPick.mock.calls.map(([p]) => p.title)).toEqual(["Your hand: open palm"]);
    expect(revoke).toHaveBeenCalledWith("blob:first.mov");
    expect(host.querySelector("p.chip")).toBeNull();
  });

  it("keeps the newer copy's busy state when an older copy fails", async () => {
    await show();
    await pickFile("first.mov");
    await failPlayback("first.mov");
    await pickFile("second.mov");
    await failPlayback("second.mov");
    await act(async () => { replies[0]({ ok: false, text: async () => "unreadable" } as Response); });
    expect(host.querySelector("p.chip")?.textContent).toContain("making a copy");
    expect(host.querySelector("p.bad")).toBeNull();
    await finishCopy(1);
    expect(onPick.mock.calls.map(([p]) => p.title)).toEqual(["second.mov"]);
  });
});
