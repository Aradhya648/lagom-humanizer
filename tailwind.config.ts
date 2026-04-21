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
        background: "#141210",
        surface: "#1E1B18",
        "surface-raised": "#272320",
        border: "#2D2924",
        "border-strong": "#3D3830",
        text: "#F2EDE8",
        muted: "#8A8078",
        faint: "#5A544E",
        accent: "#C96442",
        "accent-dim": "#A84E32",
        "accent-surface": "#2A1610",
      },
      fontFamily: {
        serif: ["Lora", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel:
          "0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.04)",
        "panel-focus":
          "0 0 0 2px rgba(201, 100, 66, 0.2), 0 1px 3px 0 rgba(0, 0, 0, 0.3)",
        btn: "0 1px 2px 0 rgba(0, 0, 0, 0.4)",
      },
      keyframes: {
        fadeWord: {
          "0%": { opacity: "0", transform: "translateY(3px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseOpacity: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
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
