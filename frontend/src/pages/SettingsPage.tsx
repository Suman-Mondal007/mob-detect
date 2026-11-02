import { Settings2, ShieldCheck, Volume2 } from "lucide-react";
import { useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { Toast } from "../components/Toast";
import { PageIntro } from "../components/PageIntro";
import { useDetectionPreferences } from "../components/DetectionPreferences";

const sensitivityDescriptions = {
  low: "Wait for at least 4 seconds above the threshold. Fewer, slower alerts.",
  balanced: "Use the server's configured confirmation time. Recommended.",
  high: "Confirm after about 0.5 seconds above the threshold. More responsive.",
};

function isSensitivity(
  value: string,
): value is "low" | "balanced" | "high" {
  return value === "low" || value === "balanced" || value === "high";
}

export function SettingsPage() {
  const { preferences, updatePreferences, setSoundEnabled } =
    useDetectionPreferences();
  const [soundError, setSoundError] = useState<string | null>(null);

  return (
    <div className="page-container settings-page">
      <PageIntro
        eyebrow="Preferences"
        title="Make alerts work for you."
        description="Adjust live-screening alerts for this browser. Your choices are saved locally."
        action={<Badge tone="brand">Live settings</Badge>}
      />

      {soundError && (
        <Toast
          tone="error"
          title="Sound setting could not be changed"
          description={soundError}
          onDismiss={() => setSoundError(null)}
        />
      )}

      <Card className="settings-card">
        <div className="settings-section-heading">
          <span className="settings-section-icon">
            <Settings2 size={19} aria-hidden="true" />
          </span>
          <div>
            <h2>Alert behavior</h2>
            <p>These controls apply to live sessions started from this browser.</p>
          </div>
        </div>

        <div className="settings-control">
          <div className="settings-control-heading">
            <label htmlFor="alert-threshold">Alert threshold</label>
            <output htmlFor="alert-threshold">
              {Math.round(preferences.alertThreshold * 100)}%
            </output>
          </div>
          <input
            id="alert-threshold"
            type="range"
            min="0.3"
            max="0.95"
            step="0.05"
            value={preferences.alertThreshold}
            aria-valuetext={`${Math.round(preferences.alertThreshold * 100)} percent`}
            aria-describedby="alert-threshold-help"
            onChange={(event) =>
              updatePreferences({
                alertThreshold: Number(event.currentTarget.value),
              })
            }
          />
          <p id="alert-threshold-help">
            Lower values trigger more easily; higher values require stronger
            model confidence. This is a screening threshold, not a measure of
            certainty.
          </p>
        </div>

        <div className="settings-control">
          <label htmlFor="alert-sensitivity">Sensitivity</label>
          <select
            id="alert-sensitivity"
            value={preferences.sensitivity}
            aria-describedby="alert-sensitivity-help"
            onChange={(event) => {
              const { value } = event.currentTarget;
              if (isSensitivity(value)) updatePreferences({ sensitivity: value });
            }}
          >
            <option value="low">Low — steadier alerts</option>
            <option value="balanced">Balanced — recommended</option>
            <option value="high">High — faster alerts</option>
          </select>
          <p id="alert-sensitivity-help">
            {sensitivityDescriptions[preferences.sensitivity]}
          </p>
        </div>

        <div className="settings-control settings-sound-control">
          <div className="settings-sound-copy">
            <span className="settings-sound-icon">
              <Volume2 size={18} aria-hidden="true" />
            </span>
            <span>
              <label htmlFor="alert-sound">Sound on alert</label>
              <small id="alert-sound-help">
                Play a brief tone when a live alert begins.
              </small>
            </span>
          </div>
          <input
            id="alert-sound"
            type="checkbox"
            role="switch"
            aria-describedby="alert-sound-help"
            checked={preferences.soundEnabled}
            onChange={(event) => {
              const { checked } = event.currentTarget;
              void setSoundEnabled(checked)
                .then(() => setSoundError(null))
                .catch((error: unknown) =>
                  setSoundError(
                    error instanceof Error
                      ? error.message
                      : "Could not change alert sound.",
                  ),
                );
            }}
          />
        </div>
      </Card>

      <div className="settings-note">
        <ShieldCheck size={17} aria-hidden="true" />
        <p>
          Uploaded-video analysis uses the threshold and duration configured by
          the server operator. Settings here change live-screening alerts only.
        </p>
      </div>
    </div>
  );
}
