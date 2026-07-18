/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        bg: { primary: '#0f0f1a', secondary: '#1a1a2e', tertiary: '#252540' },
        text: { primary: '#e8e8f0', secondary: '#9090a8' },
        accent: { DEFAULT: '#7c3aed', hover: '#8b5cf6' },
        border: '#2a2a45',
      },
      borderRadius: { DEFAULT: '12px', sm: '8px' },
    },
  },
  plugins: [],
};
