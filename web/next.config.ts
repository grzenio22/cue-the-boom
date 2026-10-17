import type { NextConfig } from "next";

// The render API (cuesync serve) runs next to the app; the browser only ever talks to this origin.
const api = process.env.CUESYNC_API_URL ?? "http://127.0.0.1:8765";

const nextConfig: NextConfig = {
  agentRules: false,
  // proxy.ts (the judging password) sees every /api request, and Next buffers what a proxy sees: 10 MB by default,
  // which cut a phone clip off mid-upload. The render API takes up to 300 MB.
  experimental: { proxyClientMaxBodySize: "320mb" },
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
};

export default nextConfig;
