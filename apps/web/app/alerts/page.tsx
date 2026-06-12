"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { getAlertHistory } from "@/lib/api";
import { aqiBadgeClass } from "@/lib/aqi-utils";
import type { AlertHistory } from "@/lib/api";

export default function AlertsPage() {
  const { isSignedIn, getToken } = useAuth();
  const router = useRouter();
  const [alerts, setAlerts] = useState<AlertHistory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isSignedIn) {
      router.push("/sign-in");
      return;
    }
    const load = async () => {
      try {
        const token = await getToken();
        if (!token) return;
        const data = await getAlertHistory(token);
        setAlerts(data);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [isSignedIn, getToken, router]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Alert History</h1>

        {loading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-14 bg-gray-100 dark:bg-gray-800 rounded-xl animate-pulse" />
            ))}
          </div>
        ) : alerts.length === 0 ? (
          <div className="text-center py-16 text-gray-500">
            <p className="text-lg font-medium mb-2">No alerts yet</p>
            <p className="text-sm">You&apos;ll see your alert history here once alerts are sent.</p>
          </div>
        ) : (
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 overflow-hidden shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800 text-xs text-gray-500 uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3 text-left">City</th>
                  <th className="px-4 py-3 text-left">Level</th>
                  <th className="px-4 py-3 text-left">Channel</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-left">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50 dark:divide-gray-800">
                {alerts.map((a) => (
                  <tr key={a.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    <td className="px-4 py-3 font-medium">{a.city}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${aqiBadgeClass(a.aqi_level)}`}>
                        {a.aqi_level}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400 capitalize">{a.channel}</td>
                    <td className="px-4 py-3">
                      <span className={`text-xs ${a.status === "delivered" ? "text-green-600" : a.status === "failed" ? "text-red-500" : "text-gray-400"}`}>
                        {a.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">
                      {new Date(a.sent_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
