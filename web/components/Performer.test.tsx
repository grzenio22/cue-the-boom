import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Performer, { type Performance } from "./Performer";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
const voice = { name: "pop.wav", url: "/pop.wav", title: "Pop", effect: "fireball" };

beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
});

const button = (text: string) => [...host.querySelectorAll("button")].find((b) => b.textContent === text)!;
const clip = () => button("The clip's own sound");
const click = (element: HTMLElement) => act(() => element.click());

function show(props: Partial<ComponentProps<typeof Performer>> = {}) {
  const onDone = vi.fn<(performance: Performance) => void>();
  act(() => root.render(<Performer plateUrl="/plate.mp4" point={[0.5, 0.5]} voices={[voice]} takes={1}
    clipSound live={false} onDone={onDone} {...props} />));
  return onDone;
}

describe("Performer replay cues", () => {
  it.each([true, false])("starts a replay with voice and prevents the clip cue (filmed=%s)", (filmed) => {
    const onDone = show({ replay: true, filmed });

    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(button("Use my microphone").disabled).toBe(false);
    expect(clip().disabled).toBe(true);
    expect(clip().title).toMatch(/voice|hold/i);
    click(clip());
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(host.textContent).not.toContain("Lock it to my clip's sound");
    expect(onDone).not.toHaveBeenCalled();
  });

  it("uses a sample voice for a replay with the GPU offline", () => {
    const onDone = show({ replay: true, filmed: true, ready: [voice.name] });
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");

    const sample = [...host.querySelectorAll("button")].find((b) => b.textContent?.startsWith("Pop"))!;
    expect(sample.disabled).toBe(false);
    click(sample);
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ cue: "voice", voice: { sample: "pop.wav" } }));
  });

  it("records a hold cue for a replay with the GPU offline", async () => {
    const onDone = show({ replay: true, filmed: true });
    expect(button("Say it").getAttribute("aria-checked")).toBe("true");
    expect(clip().disabled).toBe(true);
    click(button("Press and hold"));
    const plate = host.querySelector("video")!;
    vi.spyOn(plate, "play").mockResolvedValue(undefined);
    vi.spyOn(plate, "paused", "get").mockReturnValue(false);
    await act(async () => { button("Play the clip").click(); });
    plate.currentTime = 0.5;
    act(() => { button("Hold").dispatchEvent(new Event("pointerdown", { bubbles: true })); });
    plate.currentTime = 1;
    act(() => { button("Holding…").dispatchEvent(new Event("pointerup", { bubbles: true })); });
    act(() => { plate.dispatchEvent(new Event("ended")); });
    click(button("Use 1 hold"));

    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ cue: "hold", voice: null, holds: [[0.5, 1]] }));
  });
});
