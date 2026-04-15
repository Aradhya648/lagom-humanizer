import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Playwright can't be bundled — exclude from both edge and serverless builds
  serverExternalPackages: ["playwright", "playwright-core"],
};

export default nextConfig;
