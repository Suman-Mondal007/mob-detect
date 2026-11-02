import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "./Button";

export type ToastTone = "info" | "success" | "error";

const toneIcon: Record<ToastTone, ReactNode> = {
  info: <Info aria-hidden="true" />,
  success: <CheckCircle2 aria-hidden="true" />,
  error: <AlertCircle aria-hidden="true" />,
};

interface ToastProps {
  title: string;
  description?: string;
  tone?: ToastTone;
  onDismiss?: () => void;
}

export function Toast({
  title,
  description,
  tone = "info",
  onDismiss,
}: ToastProps) {
  return (
    <div
      className={`toast toast-${tone}`}
      role={tone === "error" ? "alert" : "status"}
    >
      <span className="toast-icon">{toneIcon[tone]}</span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-ink">{title}</p>
        {description && <p className="mt-1 text-sm text-muted">{description}</p>}
      </div>
      {onDismiss && (
        <Button
          variant="ghost"
          size="sm"
          className="toast-close"
          onClick={onDismiss}
          aria-label="Dismiss notification"
          icon={<X size={16} aria-hidden="true" />}
        />
      )}
    </div>
  );
}
