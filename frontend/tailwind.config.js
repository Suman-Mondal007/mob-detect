/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "var(--color-canvas)",
        surface: "var(--color-surface)",
        raised: "var(--color-raised)",
        ink: "var(--color-ink)",
        muted: "var(--color-muted)",
        line: "var(--color-line)",
        brand: "var(--color-brand)",
        "brand-strong": "var(--color-brand-strong)",
        "brand-soft": "var(--color-brand-soft)",
        danger: "var(--color-danger)",
        "danger-soft": "var(--color-danger-soft)",
        success: "var(--color-success)",
        "success-soft": "var(--color-success-soft)",
      },
      boxShadow: {
        card: "var(--shadow-card)",
        popover: "var(--shadow-popover)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
