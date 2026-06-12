"use client";

import { aqiBadgeClass, aqiCategoryColor } from "@/lib/aqi-utils";

interface AqiGaugeProps {
  aqi: number | null;
  category: string | null;
  city: string;
  updatedAt: string | null;
}

export function AqiGauge({ aqi, category, city, updatedAt }: AqiGaugeProps) {
  const color = aqiCategoryColor(category);
  const badgeClass = aqiBadgeClass(category);
  const displayAqi = aqi != null ? Math.round(aqi) : "—";

  // Percentage for arc (0-500 AQI scale)
  const pct = aqi != null ? Math.min(100, (aqi / 500) * 100) : 0;
  const dashArray = 2 * Math.PI * 45; // circumference of r=45
  const dashOffset = dashArray - (dashArray * pct) / 100;

  return (
    <div className="flex flex-col items-center gap-2 p-4 bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800">
      <p className="text-sm font-medium text-gray-500">{city}</p>

      {/* SVG arc gauge */}
      <svg width="120" height="70" viewBox="0 0 120 70">
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="10"
          strokeLinecap="round"
        />
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * 157} 157`}
        />
        <text x="60" y="58" textAnchor="middle" fontSize="22" fontWeight="bold" fill={color}>
          {displayAqi}
        </text>
      </svg>

      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeClass}`}>
        {category ?? "No data"}
      </span>

      {updatedAt && (
        <p className="text-xs text-gray-400">
          Updated {new Date(updatedAt).toLocaleTimeString()}
        </p>
      )}
    </div>
  );
}
