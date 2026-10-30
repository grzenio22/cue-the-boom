import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Self-hosted, both under the SIL Open Font License 1.1 (app/fonts/OFL-*.txt): the demo film's carved capitals and brush word
const display = localFont({ src: "./fonts/Cinzel.ttf", weight: "400 900", variable: "--font-display", display: "swap" });
const brush = localFont({ src: "./fonts/Shojumaru.ttf", weight: "400", variable: "--font-brush", display: "swap" });

export const metadata: Metadata = {
  title: "Cue the Boom",
  description: "Make the noise. Get the shot. Your voice picks and shapes the effect, rendered on open weights with LTX-2.5.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#15110e" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${brush.variable}`}>
      <body>{children}</body>
    </html>
  );
}
