import type { HTMLAttributes, ReactNode } from "react";

type BadgeTone = "neutral" | "brand" | "success" | "warning" | "danger";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  children: ReactNode;
}

export function Badge({
  tone = "neutral",
  children,
  className = "",
  ...props
}: BadgeProps) {
  return (
    <span className={`badge badge-${tone} ${className}`.trim()} {...props}>
      {children}
    </span>
  );
}
