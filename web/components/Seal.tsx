/** The five elements as ink seals: a ring and a few strokes each (drawn for this project, as in the demo film). */
export type Element = "fire" | "water" | "air" | "explosion" | "lightning";

export const ELEMENT_COLOR: Record<Element, string> = {
  fire: "#ff7a1a", water: "#3fa9f5", air: "#cfe3ea", explosion: "#ff4b1f", lightning: "#9db8ff",
};

const STROKES: Record<Element, string[]> = {
  fire: [
    "M50 88 C30 86 20 70 27 54 C31 63 37 65 39 61 C35 47 41 31 54 14 C55 31 67 37 70 51 C73 45 73 41 71 35 C82 47 85 64 76 77 C70 85 61 88 50 88 Z",
    "M50 80 C42 79 37 71 41 63 C44 68 48 68 49 65 C47 58 51 51 56 45 C57 54 63 59 62 68 C61 75 56 80 50 80 Z",
  ],
  water: ["M14 38 Q26 26 38 38 T62 38 T86 38", "M14 54 Q26 42 38 54 T62 54 T86 54", "M14 70 Q26 58 38 70 T62 70 T86 70"],
  air: ["M50 50 m0 -5 a5 5 0 1 1 -5 5 a10 10 0 1 1 10 10 a17 17 0 1 1 -17 -17 a26 26 0 1 1 26 26",
    "M16 30 Q34 20 52 24", "M58 80 Q72 84 86 72"],
  explosion: ["M50 10 L57 36 L82 22 L66 44 L92 52 L66 58 L80 82 L56 66 L50 92 L44 66 L20 82 L34 58 L8 52 L34 44 L18 22 L43 36 Z"],
  lightning: ["M58 10 L32 54 L49 54 L41 90 L70 42 L54 42 Z"],
};

export default function Seal({ element, size = 56, ring = 3.6, stroke = 6, className }:
  { element: Element; size?: number; ring?: number; stroke?: number; className?: string }) {
  const color = ELEMENT_COLOR[element];
  return (
    <svg className={className ? `seal ${className}` : "seal"} width={size} height={size} viewBox="0 0 100 100" aria-hidden="true"
      fill="none" stroke={color} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="50" cy="50" r="46" strokeWidth={ring} pathLength={100} />
      {STROKES[element].map((d) => <path key={d} d={d} strokeWidth={stroke} pathLength={100} />)}
    </svg>
  );
}
