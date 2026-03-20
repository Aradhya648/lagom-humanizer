/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Instrument Serif', 'Georgia', 'serif'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        lagom: {
          bg:             '#F7F5F0',
          surface:        '#FFFFFF',
          border:         '#E8E4DC',
          muted:          '#ADA89E',
          ink:            '#1A1916',
          accent:         '#2D5A3D',
          'accent-light': '#EBF2ED',
          'accent-hover': '#234830',
          coral:          '#C4573A',
          amber:          '#B8860B',
        },
      },
      fontSize: {
        'display-lg': ['3.5rem',  { lineHeight: '1.1',  letterSpacing: '-0.02em' }],
        'display-md': ['2.5rem',  { lineHeight: '1.15', letterSpacing: '-0.02em' }],
        'display-sm': ['1.75rem', { lineHeight: '1.2',  letterSpacing: '-0.01em' }],
      },
      boxShadow: {
        'lagom-sm': '0 1px 3px 0 rgba(26,25,22,0.06), 0 1px 2px -1px rgba(26,25,22,0.06)',
        'lagom-md': '0 4px 12px 0 rgba(26,25,22,0.08), 0 2px 4px -2px rgba(26,25,22,0.05)',
        'lagom-lg': '0 8px 32px 0 rgba(26,25,22,0.10), 0 4px 8px -4px rgba(26,25,22,0.06)',
      },
      borderRadius: {
        'lagom':    '10px',
        'lagom-lg': '16px',
        'lagom-xl': '24px',
      },
      animation: {
        'fade-up':    'fadeUp 0.5s ease-out forwards',
        'fade-in':    'fadeIn 0.3s ease-out forwards',
        'shimmer':    'shimmer 1.6s ease-in-out infinite',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeUp: {
          '0%':   { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        shimmer: {
          '0%, 100%': { opacity: '0.4' },
          '50%':      { opacity: '1' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0.6' },
        },
      },
    },
  },
  plugins: [],
}
