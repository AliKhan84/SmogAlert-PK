"use client";

interface PollutantBarProps {
  label: string;
  value: number | null;
  max: number;
  unit?: string;
}

function Bar({ label, value, max, unit = "µg/m³" }: PollutantBarProps) {
  const pct = value != null ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="w-12 text-xs text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-2">
        <div
          className="h-2 rounded-full bg-teal-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-24 text-right text-xs text-gray-600 dark:text-gray-300 shrink-0">
        {value != null ? `${value.toFixed(1)} ${unit}` : "—"}
      </span>
    </div>
  );
}

interface PollutantPanelProps {
  pm25: number | null;
  pm10: number | null;
  no2: number | null;
  so2?: number | null;
  co?: number | null;
  o3?: number | null;
}

export function PollutantPanel({ pm25, pm10, no2, so2, co, o3 }: PollutantPanelProps) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">Pollutant Breakdown</h3>
      <Bar label="PM2.5" value={pm25} max={250} />
      <Bar label="PM10" value={pm10} max={430} />
      <Bar label="NO₂" value={no2} max={200} />
      <Bar label="SO₂" value={so2 ?? null} max={350} />
      <Bar label="CO" value={co ?? null} max={10} unit="mg/m³" />
      <Bar label="O₃" value={o3 ?? null} max={200} />
    </div>
  );
}
