"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import {
  getProfile,
  getPreferences,
  updatePreferences,
  sendTestAlert,
  addLocation,
  deleteLocation,
} from "@/lib/api";
import type { UserProfile, AlertPreferences } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { PAKISTAN_CITIES } from "@/lib/aqi-utils";

const AQI_THRESHOLDS = ["Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"];

export default function SettingsPage() {
  const { isSignedIn, getToken } = useAuth();
  const router = useRouter();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [prefs, setPrefs] = useState<AlertPreferences | null>(null);
  const [saving, setSaving] = useState(false);
  const [testSent, setTestSent] = useState(false);

  useEffect(() => {
    if (!isSignedIn) {
      router.push("/sign-in");
      return;
    }
    const load = async () => {
      const token = await getToken();
      if (!token) return;
      const [p, pr] = await Promise.all([getProfile(token), getPreferences(token)]);
      setProfile(p);
      setPrefs(pr);
    };
    load();
  }, [isSignedIn, getToken, router]);

  const handleSavePrefs = async () => {
    if (!prefs) return;
    setSaving(true);
    try {
      const token = await getToken();
      if (!token) return;
      const updated = await updatePreferences(token, prefs);
      setPrefs(updated);
    } finally {
      setSaving(false);
    }
  };

  const handleAddCity = async (city: string) => {
    const token = await getToken();
    if (!token) return;
    const isPrimary = profile?.locations.length === 0;
    await addLocation(token, city, isPrimary);
    const updated = await getProfile(token);
    setProfile(updated);
  };

  const handleRemoveCity = async (locationId: number) => {
    const token = await getToken();
    if (!token) return;
    await deleteLocation(token, locationId);
    const updated = await getProfile(token);
    setProfile(updated);
  };

  const handleTestAlert = async () => {
    const token = await getToken();
    if (!token) return;
    await sendTestAlert(token);
    setTestSent(true);
    setTimeout(() => setTestSent(false), 3000);
  };

  if (!profile || !prefs) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-teal-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  const monitoredCities = profile.locations.map((l) => l.city);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <div className="max-w-2xl mx-auto px-4 py-8 space-y-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>

        {/* Profile */}
        <section className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-5 shadow-sm">
          <h2 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">Profile</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">{profile.email}</p>
          {profile.name && <p className="text-sm text-gray-600 dark:text-gray-400">{profile.name}</p>}
        </section>

        {/* Cities */}
        <section className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-5 shadow-sm">
          <h2 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">Monitored Cities</h2>
          <div className="flex flex-wrap gap-2 mb-4">
            {profile.locations.map((loc) => (
              <span
                key={loc.id}
                className="inline-flex items-center gap-2 bg-teal-50 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300 px-3 py-1 rounded-full text-sm"
              >
                {loc.city}
                {loc.is_primary && <span className="text-xs opacity-70">(primary)</span>}
                <button
                  onClick={() => handleRemoveCity(loc.id)}
                  className="text-teal-500 hover:text-red-500 ml-1"
                  aria-label={`Remove ${loc.city}`}
                >
                  x
                </button>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {PAKISTAN_CITIES.filter((c) => !monitoredCities.includes(c)).map((city) => (
              <button
                key={city}
                onClick={() => handleAddCity(city)}
                className="text-xs px-3 py-1 rounded-full border border-gray-200 dark:border-gray-700 hover:border-teal-400 hover:text-teal-600 transition-colors"
              >
                + {city}
              </button>
            ))}
          </div>
        </section>

        {/* Alert preferences */}
        <section className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-5 shadow-sm">
          <h2 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">Alert Preferences</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Alert threshold</label>
              <select
                value={prefs.aqi_threshold}
                onChange={(e) => setPrefs({ ...prefs, aqi_threshold: e.target.value })}
                className="w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
              >
                {AQI_THRESHOLDS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Language</label>
              <select
                value={prefs.language}
                onChange={(e) => setPrefs({ ...prefs, language: e.target.value })}
                className="w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
              >
                <option value="en">English</option>
                <option value="ur">اردو (Urdu)</option>
              </select>
            </div>
          </div>

          <div className="flex gap-3 mt-5">
            <Button onClick={handleSavePrefs} disabled={saving}>
              {saving ? "Saving..." : "Save preferences"}
            </Button>
            <Button
              variant="secondary"
              onClick={handleTestAlert}
              disabled={testSent}
            >
              {testSent ? "Test sent!" : "Send test alert"}
            </Button>
          </div>
        </section>
      </div>
    </div>
  );
}
