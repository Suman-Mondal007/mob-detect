import type { HTMLAttributes } from "react";

interface SpinnerProps extends HTMLAttributes<HTMLSpanElement> {
  label?: string;
  size?: "sm" | "md" | "lg";
}

export function Spinner({
  label = "Loading",
  size = "md",
  className = "",
  ...props
}: SpinnerProps) {
  return (
    <span
      className={`spinner spinner-${size} ${className}`.trim()}
      role="status"
      aria-label={label}
      {...props}
    />
  );
}
