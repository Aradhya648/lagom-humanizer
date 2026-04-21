import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#050810",
        surface: "#0C1220",
        "surface-raised": "#121A2A",
        border: "#1A2840",
        "border-strong": "#243850",
        text: "#E0F0FF",
        muted: "#608090",
        faint: "#304858",
        accent: "#00D4AA",
        "accent-dim": "#00A884",
        "accent-surface": "#021815",
        "gradient-from": "#00D4AA",
        "gradient-to": "#C084FC",
      },
      fontFamily: {
        serif: ["Lora", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel:
          "0 1px 3px 0 rgba(0,0,0,0.6), 0 0 0 1px rgba(0,212,170,0.07)",
        "panel-active":
          "0 1px 3px 0 rgba(0,0,0,0.6), 0 0 0 1px rgba(0,212,170,0.18), inset 0 0 0 1px rgba(0,212,170,0.07)",
        btn: "0 1px 3px 0 rgba(0,0,0,0.5), 0 0 18px 0 rgba(0,212,170,0.3)",
      },
      keyframes: {
        fadeWord: {
          "0%": { opacity: "0", transform: "translateY(3px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseOpacity: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        "fade-word": "fadeWord 0.22s ease forwards",
        "pulse-opacity": "pulseOpacity 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
