"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastPoint } from "@/lib/api";
import { pm25ToCategory, aqiCategoryColor } from "@/lib/aqi-utils";

interface ForecastChartProps {
  points: ForecastPoint[];
  city: string;
}

export function ForecastChart({ points, city }: ForecastChartProps) {
  const data = points.map((p) => ({
    time: new Date(p.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    pm25: +p.pm25_predicted.toFixed(1),
    lower: +p.lower_ci.toFixed(1),
    upper: +p.upper_ci.toFixed(1),
    category: p.aqi_category ?? pm25ToCategory(p.pm25_predicted),
  }));

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
        24-Hour PM2.5 Forecast — {city}
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <defs>
            <linearGradient id="pm25Gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0d9488" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#0d9488" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#94a3b8" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} tickLine={false} />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} unit=" µg" />
          <Tooltip
            formatter={(val, name) => {
              const labels: Record<string, string> = {
                pm25: "PM2.5",
                upper: "Upper CI",
                lower: "Lower CI",
              };
              return [`${Number(val).toFixed(1)} µg/m³`, labels[String(name)] ?? String(name)];
            }}
          />
          {/* Confidence interval band */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#ciGradient)"
            fillOpacity={1}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#fff"
            fillOpacity={1}
          />
          {/* Main forecast line */}
          <Area
            type="monotone"
            dataKey="pm25"
            stroke="#0d9488"
            strokeWidth={2}
            fill="url(#pm25Gradient)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 mt-2">Shaded area = 80% confidence interval</p>
    </div>
  );
}
