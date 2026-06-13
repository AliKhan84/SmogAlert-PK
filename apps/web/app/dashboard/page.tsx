"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { useAuth } from "@clerk/nextjs";
import { AqiGauge } from "@/components/dashboard/AqiGauge";
import { ForecastChart } from "@/components/dashboard/ForecastChart";
import { PollutantPanel } from "@/components/dashboard/PollutantBar";
import { getCurrentAqi, getCityForecast, getCityHistory } from "@/lib/api";
import { aqiBadgeClass, PAKISTAN_CITIES } from "@/lib/aqi-utils";
import { useAqiWebSocket } from "@/lib/aqi-ws";
import type { CurrentAqi, ForecastResponse, AqiReading } from "@/lib/api";

// Map must load client-side only (Leaflet requires window)
const PakistanMap = dynamic(
  () => import("@/components/dashboard/PakistanMap").then((m) => m.PakistanMap),
  { ssr: false }
);

export default function DashboardPage() {
  const { isSignedIn } = useAuth();
  const [aqiData, setAqiData] = useState<CurrentAqi[]>([]);
  const [selectedCity, setSelectedCity] = useState("Lahore");
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [history, setHistory] = useState<AqiReading[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getCurrentAqi();
        setAqiData(data);
        setLastUpdated(new Date());
      } catch (err) {
        console.error("Failed to load AQI data", err);
      } finally {
        setLoading(false);
      }
    };
    load();
    // Auto-refresh every 5 minutes as fallback
    const interval = setInterval(load, 5 * 60 * 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  // Live updates from Railway WebSocket — merges aqi/category/pm25 without
  // clobbering pm10/no2/source that only come from the REST endpoint.
  useAqiWebSocket((wsData) => {
    setAqiData((prev) =>
      prev.map((existing) => {
        const update = wsData.find((d) => d.city === existing.city);
        if (!update) return existing;
        return {
          ...existing,
          aqi: update.aqi,
          category: update.category,
          pm25: update.pm25,
          updated_at: update.updated_at,
        };
      })
    );
    setLastUpdated(new Date());
  });

  useEffect(() => {
    const loadCity = async () => {
      try {
        const [fc, hist] = await Promise.all([
          getCityForecast(selectedCity),
          getCityHistory(selectedCity, 7),
        ]);
        setForecast(fc);
        setHistory(hist);
      } catch (err) {
        console.error("Failed to load city data", err);
      }
    };
    loadCity();
  }, [selectedCity]);

  const selected = aqiData.find((r) => r.city === selectedCity);
  const anomalyCount = history.filter((r) => r.is_anomaly).length;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      {/* Top bar */}
      <div className="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800 px-4 py-3 flex items-center justify-between">
        <h1 className="font-bold text-teal-700 dark:text-teal-400">SmogAlert PK</h1>
        <div className="flex items-center gap-4 text-sm text-gray-500">
          {lastUpdated && (
            <span>Updated {lastUpdated.toLocaleTimeString()}</span>
          )}
          {isSignedIn && (
            <a href="/settings" className="text-teal-600 hover:underline">Settings</a>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* City AQI overview */}
        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {PAKISTAN_CITIES.map((c) => (
              <div key={c} className="h-36 bg-gray-100 dark:bg-gray-800 rounded-2xl animate-pulse" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {aqiData.map((r) => (
              <button
                key={r.city}
                onClick={() => setSelectedCity(r.city)}
                className={`text-left rounded-2xl ring-2 transition-all ${selectedCity === r.city ? "ring-teal-500" : "ring-transparent"}`}
              >
                <AqiGauge
                  city={r.city}
                  aqi={r.aqi}
                  category={r.category}
                  updatedAt={r.updated_at}
                />
              </button>
            ))}
          </div>
        )}

        {/* Map + pollutants */}
        <div className="grid lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <PakistanMap
              readings={aqiData}
              selectedCity={selectedCity}
              onCitySelect={setSelectedCity}
            />
          </div>
          <div>
            {selected ? (
              <PollutantPanel
                pm25={selected.pm25}
                pm10={selected.pm10}
                no2={selected.no2}
              />
            ) : (
              <div className="h-64 bg-gray-100 dark:bg-gray-800 rounded-2xl animate-pulse" />
            )}
          </div>
        </div>

        {/* Forecast chart */}
        {forecast && <ForecastChart points={forecast.points} city={selectedCity} />}

        {/* Anomaly summary */}
        {history.length > 0 && (
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-4 shadow-sm">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Recent Anomalies — {selectedCity} (last 7 days)
            </h3>
            <p className="text-2xl font-bold text-red-500 mb-2">{anomalyCount}</p>
            <p className="text-xs text-gray-500">anomalous readings detected out of {history.length} total</p>
            <div className="mt-3 space-y-1 max-h-40 overflow-y-auto">
              {history
                .filter((r) => r.is_anomaly)
                .slice(-10)
                .reverse()
                .map((r, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className={`px-1.5 py-0.5 rounded text-xs ${aqiBadgeClass(r.aqi_category)}`}>
                      {r.aqi_category ?? "Unknown"}
                    </span>
                    <span className="text-gray-500">
                      {new Date(r.timestamp).toLocaleString()} — PM2.5: {r.pm25?.toFixed(1) ?? "—"}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
