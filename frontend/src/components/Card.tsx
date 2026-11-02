import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLElement> {
  children: ReactNode;
  interactive?: boolean;
}

export function Card({
  children,
  interactive = false,
  className = "",
  ...props
}: CardProps) {
  return (
    <section
      className={`card ${interactive ? "card-interactive" : ""} ${className}`.trim()}
      {...props}
    >
      {children}
    </section>
  );
}
