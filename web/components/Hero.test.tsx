import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import Hero from "./Hero";

describe("Hero", () => {
  it("states the promise, names the five elements and sends the main action to the studio", () => {
    const html = renderToStaticMarkup(<Hero />);
    expect(html).toContain("Render it once.");
    expect(html).toContain("No regeneration");
    for (const element of ["Fire", "Water", "Air", "Explosion", "Lightning"]) expect(html).toContain(`>${element}<`);
    expect(html).toMatch(/<a class="btn hero-cta" href="#studio">/);
    expect(html).toContain('src="/hero/reel.mp4"');
  });
});
