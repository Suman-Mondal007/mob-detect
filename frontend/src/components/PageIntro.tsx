import type { ReactNode } from "react";

interface PageIntroProps {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}

export function PageIntro({
  eyebrow,
  title,
  description,
  action,
}: PageIntroProps) {
  return (
    <div className="page-intro">
      <div className="min-w-0">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="page-title">{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {action && <div className="page-intro-action">{action}</div>}
    </div>
  );
}
