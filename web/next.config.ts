import type { NextConfig } from "next";

// The render API (cuesync serve) runs next to the app; the browser only ever talks to this origin.
const api = process.env.CUESYNC_API_URL ?? "http://127.0.0.1:8765";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
};

export default nextConfig;
