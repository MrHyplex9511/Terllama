/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0f0f1a',
          secondary: '#1a1a2e',
          tertiary: '#252540',
        },
        content: {
          DEFAULT: '#e8e8f0',
          muted: '#9090a8',
        },
        brand: {
          DEFAULT: '#7C3AED',
          hover: '#8B5CF6',
          light: '#A78BFA',
        },
        gpu: {
          DEFAULT: '#06B6D4',
          hover: '#22D3EE',
        },
        success: '#22C55E',
        warning: '#F59E0B',
        danger: '#EF4444',
        border: '#2a2a45',
      },
      borderRadius: {
        DEFAULT: '12px',
        sm: '8px',
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.2s ease-out',
        'pulse-slow': 'pulse 3s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};
