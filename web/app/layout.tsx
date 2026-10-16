import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cue the Boom",
  description: "Make the noise. Get the shot. Your voice picks and shapes the effect, rendered on open weights with LTX-2.5.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0b0b0f" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
