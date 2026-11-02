import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VideoAlertRange, VideoResult } from "../api/client";

interface VideoTimelineChartProps {
  result: VideoResult;
  onSeek: (seconds: number) => void;
}

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainingSeconds = safeSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`
    : `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

export function VideoTimelineChart({
  result,
  onSeek,
}: VideoTimelineChartProps) {
  function seekFromChart(index: number | string | null | undefined) {
    if (index === null || index === undefined || index === "") return;
    const pointIndex = Number(index);
    if (!Number.isInteger(pointIndex)) return;
    const point = result.timeline[pointIndex];
    if (point) onSeek(point.time_seconds);
  }

  function renderPoint(props: {
    cx?: number;
    cy?: number;
    index: number;
  }) {
    const point = result.timeline[props.index];
    if (props.cx === undefined || props.cy === undefined || !point) return null;
    const seek = () => onSeek(point.time_seconds);
    return (
      <circle
        cx={props.cx}
        cy={props.cy}
        r={5}
        fill="var(--color-brand)"
        stroke="var(--color-surface)"
        strokeWidth={2}
        role="button"
        tabIndex={0}
        aria-label={`Seek to ${formatTime(point.time_seconds)}, probability ${Math.round(point.violence_probability * 100)} percent`}
        style={{ cursor: "pointer" }}
        onClick={(event) => {
          event.stopPropagation();
          seek();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            seek();
          }
        }}
      />
    );
  }

  return (
    <div
      className="upload-chart"
      role="group"
      aria-label="Violence probability over video time. Click the chart to seek the preview."
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={result.timeline}
          margin={{ top: 14, right: 15, left: -12, bottom: 3 }}
          onClick={(state) => seekFromChart(state.activeTooltipIndex)}
        >
          {result.alert_ranges.map((range: VideoAlertRange, index) => (
            <ReferenceArea
              key={`${range.start_seconds}-${range.end_seconds}-${index}`}
              x1={range.start_seconds}
              x2={range.end_seconds}
              y1={0}
              y2={1}
              fill="var(--color-danger)"
              fillOpacity={0.1}
              ifOverflow="extendDomain"
            />
          ))}
          <CartesianGrid
            stroke="var(--color-line)"
            strokeDasharray="3 5"
            vertical={false}
          />
          <XAxis
            dataKey="time_seconds"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value: number) => formatTime(value)}
            tick={{ fill: "var(--color-muted)", fontSize: 10 }}
            axisLine={{ stroke: "var(--color-line)" }}
            tickLine={false}
            minTickGap={28}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
            tick={{ fill: "var(--color-muted)", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={43}
          />
          <Tooltip
            formatter={(value) => [
              `${Math.round(Number(value) * 100)}%`,
              "Violence probability",
            ]}
            labelFormatter={(value) => formatTime(Number(value))}
            contentStyle={{
              border: "1px solid var(--color-line)",
              borderRadius: 10,
              background: "var(--color-surface)",
              color: "var(--color-ink)",
              fontSize: 11,
            }}
          />
          <Line
            type="monotone"
            dataKey="violence_probability"
            name="Violence probability"
            stroke="var(--color-brand)"
            strokeWidth={2.5}
            activeDot={{
              r: 5,
              fill: "var(--color-brand)",
              stroke: "var(--color-surface)",
              strokeWidth: 2,
            }}
            dot={renderPoint}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
