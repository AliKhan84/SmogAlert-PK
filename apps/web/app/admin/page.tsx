"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

interface ModelVersion {
  id: number;
  model_type: string;
  city: string | null;
  mae: number | null;
  r2_score: number | null;
  trained_at: string;
  promoted: boolean;
  r2_key: string | null;
}

interface IngestionStatus {
  city: string;
  source: string;
  last_seen: string | null;
  total_rows: number;
}

async function apiFetch<T>(path: string, token: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...opts,
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json", ...opts.headers },
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

export default function AdminPage() {
  const { getToken } = useAuth();
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [ingestion, setIngestion] = useState<IngestionStatus[]>([]);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      const token = await getToken();
      if (!token) return;
      const [mv, ing] = await Promise.all([
        apiFetch<ModelVersion[]>("/admin/model-status", token),
        apiFetch<IngestionStatus[]>("/admin/ingestion-status", token),
      ]);
      setModels(mv);
      setIngestion(ing);
      setError("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load admin data — are you an admin?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const triggerRetrain = async (city?: string) => {
    setRetraining(true);
    setRetrainMsg("");
    try {
      const token = await getToken();
      if (!token) return;
      const params = city ? `?city=${city}` : "";
      const res = await apiFetch<{ status: string; task_id: string }>(`/admin/retrain${params}`, token, { method: "POST" });
      setRetrainMsg(`Retrain queued — task ${res.task_id}`);
      // Refresh model list after a short delay
      setTimeout(load, 3000);
    } catch (e: unknown) {
      setRetrainMsg(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setRetraining(false);
    }
  };

  const sourceColour = (source: string) => {
    const map: Record<string, string> = {
      openaq: "bg-blue-100 text-blue-700",
      waqi: "bg-teal-100 text-teal-700",
      openmeteo: "bg-green-100 text-green-700",
      kaggle: "bg-purple-100 text-purple-700",
      epa: "bg-orange-100 text-orange-700",
    };
    return map[source.toLowerCase()] ?? "bg-gray-100 text-gray-700";
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 p-6">
      <div className="max-w-6xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Admin Panel</h1>
            <p className="text-sm text-gray-500 mt-1">SmogAlert PK — internal dashboard</p>
          </div>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={load} disabled={loading}>Refresh</Button>
            <Button onClick={() => triggerRetrain()} disabled={retraining}>
              {retraining ? "Queuing..." : "Retrain All Models"}
            </Button>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">{error}</div>
        )}
        {retrainMsg && (
          <div className="bg-teal-50 border border-teal-200 rounded-xl p-4 text-teal-700 text-sm">{retrainMsg}</div>
        )}

        {/* Data Ingestion Status */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800">
            <h2 className="font-semibold text-gray-800 dark:text-gray-200">Data Ingestion Status</h2>
            <p className="text-xs text-gray-500 mt-0.5">Last successful scrape per city and source</p>
          </div>
          {loading ? (
            <div className="p-6 space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-10 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b border-gray-100 dark:border-gray-800">
                    <th className="px-6 py-3 font-medium">City</th>
                    <th className="px-4 py-3 font-medium">Source</th>
                    <th className="px-4 py-3 font-medium">Last Seen</th>
                    <th className="px-4 py-3 font-medium">Total Rows</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {ingestion.map((row, i) => {
                    const lastSeen = row.last_seen ? new Date(row.last_seen) : null;
                    const hoursAgo = lastSeen ? (Date.now() - lastSeen.getTime()) / 3600000 : null;
                    const isStale = hoursAgo !== null && hoursAgo > 3;
                    return (
                      <tr key={i} className="border-b border-gray-50 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                        <td className="px-6 py-3 font-medium text-gray-900 dark:text-white">{row.city}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${sourceColour(row.source)}`}>
                            {row.source}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                          {lastSeen ? lastSeen.toLocaleString() : "—"}
                        </td>
                        <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                          {row.total_rows.toLocaleString()}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            isStale ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                          }`}>
                            {isStale ? `Stale (${hoursAgo?.toFixed(0)}h)` : "Fresh"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                  {ingestion.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-6 py-8 text-center text-sm text-gray-400">
                        No ingestion data yet — run the scraper or backfill Kaggle data.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Model Version History */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between">
            <div>
              <h2 className="font-semibold text-gray-800 dark:text-gray-200">Model Version History</h2>
              <p className="text-xs text-gray-500 mt-0.5">Prophet MAE per city — lower is better</p>
            </div>
            <div className="flex gap-2">
              {["Islamabad", "Karachi", "Lahore", "Peshawar", "Quetta"].map((city) => (
                <button
                  key={city}
                  onClick={() => triggerRetrain(city)}
                  disabled={retraining}
                  className="text-xs px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-teal-50 hover:border-teal-300 dark:hover:bg-teal-900/20 transition-colors disabled:opacity-50"
                >
                  Retrain {city}
                </button>
              ))}
            </div>
          </div>
          {loading ? (
            <div className="p-6 space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-10 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b border-gray-100 dark:border-gray-800">
                    <th className="px-6 py-3 font-medium">City</th>
                    <th className="px-4 py-3 font-medium">Type</th>
                    <th className="px-4 py-3 font-medium">MAE (µg/m³)</th>
                    <th className="px-4 py-3 font-medium">Trained At</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                    <th className="px-4 py-3 font-medium">R2 Key</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.id} className="border-b border-gray-50 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="px-6 py-3 font-medium text-gray-900 dark:text-white">{m.city ?? "all"}</td>
                      <td className="px-4 py-3 text-gray-500">{m.model_type}</td>
                      <td className="px-4 py-3 font-mono">
                        {m.mae !== null ? (
                          <span className={m.mae < 20 ? "text-green-600" : m.mae < 40 ? "text-yellow-600" : "text-red-600"}>
                            {m.mae.toFixed(2)}
                          </span>
                        ) : "—"}
                      </td>
                      <td className="px-4 py-3 text-gray-500">
                        {new Date(m.trained_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          m.promoted ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"
                        }`}>
                          {m.promoted ? "Promoted" : "Rejected"}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-gray-400 truncate max-w-xs">
                        {m.r2_key ?? "local only"}
                      </td>
                    </tr>
                  ))}
                  {models.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-6 py-8 text-center text-sm text-gray-400">
                        No model versions yet — trigger a retrain above.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
