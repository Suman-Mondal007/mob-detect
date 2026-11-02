import {
  Activity,
  Camera,
  Database,
  HeartPulse,
  ScanEye,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { PageIntro } from "../components/PageIntro";

const workflow = [
  {
    icon: Camera,
    title: "Capture",
    description:
      "Live preview stays in your browser. Small JPEG frames are sent to the configured analysis server; uploaded videos are sampled for analysis.",
  },
  {
    icon: ScanEye,
    title: "Analyze",
    description:
      "The existing trained classifier estimates violence probability for each sampled frame. A temporal smoother reduces one-frame spikes.",
  },
  {
    icon: Activity,
    title: "Review",
    description:
      "When configured thresholds are met, Mob-Alert records an event for a person to review. It does not decide what happened.",
  },
];

const principles = [
  {
    icon: ScanEye,
    title: "A screening aid",
    description:
      "Mob-Alert looks for patterns in video frames that may warrant human review.",
  },
  {
    icon: HeartPulse,
    title: "Not a safety system",
    description:
      "This prototype is not validated for emergency response or operational safety decisions.",
  },
  {
    icon: ShieldCheck,
    title: "Human judgment matters",
    description:
      "Models can miss events or produce false alarms. Context and human review remain essential.",
  },
];

export function AboutPage() {
  return (
    <div className="page-container">
      <PageIntro
        eyebrow="About the project"
        title="Built to explore, not to decide."
        description="Mob-Alert is a college project exploring how computer vision might help screen video for possible violence."
        action={<Badge tone="brand">Prototype</Badge>}
      />

      <Card className="about-callout">
        <div className="about-callout-icon">
          <HeartPulse size={20} aria-hidden="true" />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-ink">How it works</h2>
          <p className="mt-1 text-sm leading-6 text-muted">
            Mob-Alert reuses its existing trained model and temporal alert logic.
            It does not retrain the model or identify people.
          </p>
        </div>
      </Card>

      <div className="about-workflow-grid">
        {workflow.map(({ icon: Icon, title, description }, index) => (
          <Card key={title} className="principle-card">
            <span className="principle-number">0{index + 1}</span>
            <span className="principle-icon">
              <Icon size={19} aria-hidden="true" />
            </span>
            <h2 className="mt-5 text-sm font-semibold text-ink">{title}</h2>
            <p className="mt-2 text-sm leading-6 text-muted">{description}</p>
          </Card>
        ))}
      </div>

      <section className="about-section" aria-labelledby="limitations-title">
        <div className="about-section-heading">
          <h2 id="limitations-title">Limitations to keep in mind</h2>
          <p>Model output can be wrong, incomplete, or out of context.</p>
        </div>
        <div className="principle-grid">
          {principles.map(({ icon: Icon, title, description }, index) => (
            <Card key={title} className="principle-card">
              <span className="principle-number">0{index + 1}</span>
              <span className="principle-icon">
                <Icon size={19} aria-hidden="true" />
              </span>
              <h2 className="mt-5 text-sm font-semibold text-ink">{title}</h2>
              <p className="mt-2 text-sm leading-6 text-muted">{description}</p>
            </Card>
          ))}
        </div>
      </section>

      <Card className="about-privacy-card">
        <span className="about-privacy-icon">
          <Database size={19} aria-hidden="true" />
        </span>
        <div>
          <h2>Privacy and video handling</h2>
          <p>
            Uploaded video is processed from a temporary file and deleted when
            analysis finishes. Live frames are analyzed as they arrive; a full
            video recording is not stored by this app. The browser sends
            downscaled frames to the configured backend, so live video is not
            entirely local.
          </p>
          <p>
            When an alert is raised, event metadata and a single JPEG snapshot
            are retained in the backend&apos;s SQLite database and snapshot
            folder so the Alerts page can show history. The operator controls
            those files and their retention. Avoid sharing sensitive footage
            unless you have permission.
          </p>
        </div>
      </Card>
    </div>
  );
}
