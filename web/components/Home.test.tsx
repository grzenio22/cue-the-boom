import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import Home from "@/app/page";

vi.mock("@/components/Studio", () => ({ default: () => <button>Perform this fire with your voice</button> }));
vi.mock("@/components/Again", () => ({ default: () => <section>Three performances</section> }));
vi.mock("@/components/Hook", () => ({ default: () => <section>Other finished shots</section> }));
vi.mock("@/components/Gallery", () => ({ default: () => <section>Gallery</section> }));

describe("Home creation order", () => {
  it("puts the creation action before the showcase videos in document and mobile reading order", () => {
    const html = renderToStaticMarkup(<Home />);
    const action = html.indexOf("Perform this fire with your voice");
    expect(action).toBeGreaterThan(0);
    for (const showcase of ["Three performances", "Other finished shots", "Gallery"]) {
      expect(action).toBeLessThan(html.indexOf(showcase));
    }
  });
});
