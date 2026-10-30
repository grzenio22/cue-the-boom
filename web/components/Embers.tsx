"use client";

import { useEffect, useRef } from "react";

type Mote = { x: number; y: number; vx: number; vy: number; r: number; life: number; age: number; hue: number; wind: boolean };

/** The night air behind the page: embers rising off the lanterns and, now and then, a thread of wind crossing.
 * One small canvas, drawn only while the tab is visible; nothing at all for a viewer who asks for less motion. */
export default function Embers() {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let w = 0, h = 0, frame = 0, last = performance.now();
    const motes: Mote[] = [];
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      w = window.innerWidth; h = window.innerHeight;
      el.width = Math.round(w * dpr); el.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const spawn = (anywhere: boolean): Mote => {
      const wind = Math.random() < 0.12;
      return wind
        ? { x: -80, y: h * (0.15 + Math.random() * 0.7), vx: 90 + Math.random() * 80, vy: -6 + Math.random() * 12, r: 40 + Math.random() * 70,
          life: 4 + Math.random() * 3, age: 0, hue: 195, wind }
        : { x: Math.random() * w, y: anywhere ? Math.random() * h : h + 10, vx: -8 + Math.random() * 16, vy: -(14 + Math.random() * 30),
          r: 0.8 + Math.random() * 1.8, life: 6 + Math.random() * 8, age: anywhere ? Math.random() * 6 : 0, hue: 18 + Math.random() * 26, wind };
    };
    resize();
    const count = () => Math.min(70, Math.round((w * h) / 22000));
    for (let i = 0; i < count(); i++) motes.push(spawn(true));

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < motes.length; i++) {
        const m = motes[i];
        m.age += dt;
        m.x += (m.vx + (m.wind ? 0 : Math.sin(now / 900 + i) * 9)) * dt;
        m.y += m.vy * dt;
        const t = m.age / m.life;
        if (t >= 1 || m.y < -20 || m.x > w + 160) { motes[i] = spawn(false); continue; }
        const fade = Math.sin(Math.PI * t);
        if (m.wind) {
          ctx.strokeStyle = `hsla(${m.hue}, 40%, 85%, ${0.07 * fade})`;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(m.x - m.r, m.y);
          ctx.bezierCurveTo(m.x - m.r * 0.4, m.y - 10, m.x + m.r * 0.2, m.y + 8, m.x + m.r, m.y - 4);
          ctx.stroke();
        } else {
          const flicker = 0.65 + 0.35 * Math.sin(now / 120 + i * 7);
          const g = ctx.createRadialGradient(m.x, m.y, 0, m.x, m.y, m.r * 4);
          g.addColorStop(0, `hsla(${m.hue + 15}, 100%, 75%, ${0.85 * fade * flicker})`);
          g.addColorStop(0.3, `hsla(${m.hue}, 100%, 55%, ${0.35 * fade * flicker})`);
          g.addColorStop(1, "hsla(20, 100%, 50%, 0)");
          ctx.fillStyle = g;
          ctx.beginPath(); ctx.arc(m.x, m.y, m.r * 4, 0, Math.PI * 2); ctx.fill();
        }
      }
      while (motes.length < count()) motes.push(spawn(false));
      if (motes.length > count()) motes.length = count();
      frame = requestAnimationFrame(draw);
    };
    const visibility = () => {
      cancelAnimationFrame(frame);
      if (!document.hidden) { last = performance.now(); frame = requestAnimationFrame(draw); }
    };
    frame = requestAnimationFrame(draw);
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);

  return <canvas ref={canvas} className="embers" aria-hidden="true" />;
}
