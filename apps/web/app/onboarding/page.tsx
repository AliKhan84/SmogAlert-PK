"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { PAKISTAN_CITIES } from "@/lib/aqi-utils";
import { addLocation, updatePreferences } from "@/lib/api";

const STEPS = ["Choose City", "Alert Level", "Done"] as const;
const AQI_THRESHOLDS = ["Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"];

export default function OnboardingPage() {
  const { getToken } = useAuth();
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [city, setCity] = useState("");
  const [threshold, setThreshold] = useState("Unhealthy");
  const [submitting, setSubmitting] = useState(false);

  const handleFinish = async () => {
    setSubmitting(true);
    try {
      const token = await getToken();
      if (token) {
        // Best-effort: save preferences to API; failures are non-blocking
        await Promise.allSettled([
          addLocation(token, city, true),
          updatePreferences(token, { aqi_threshold: threshold, channels: "email" }),
        ]);
      }
    } catch {
      // Ignore API errors — user can configure in Settings
    } finally {
      setSubmitting(false);
    }
    router.push("/dashboard");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-white dark:from-gray-950 dark:to-gray-900 flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-white dark:bg-gray-900 rounded-2xl shadow-lg border border-gray-100 dark:border-gray-800 p-8">
        {/* Progress */}
        <div className="flex gap-2 mb-8">
          {STEPS.map((s, i) => (
            <div
              key={s}
              className={`flex-1 h-1.5 rounded-full transition-colors ${i <= step ? "bg-teal-500" : "bg-gray-200 dark:bg-gray-700"}`}
            />
          ))}
        </div>

        {step === 0 && (
          <>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              Which city do you want to monitor?
            </h2>
            <p className="text-sm text-gray-500 mb-6">You can add more cities later in settings.</p>
            <div className="grid grid-cols-2 gap-3">
              {PAKISTAN_CITIES.map((c) => (
                <button
                  key={c}
                  onClick={() => setCity(c)}
                  className={`py-3 px-4 rounded-xl border text-sm font-medium transition-all ${
                    city === c
                      ? "border-teal-500 bg-teal-50 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300"
                      : "border-gray-200 dark:border-gray-700 hover:border-teal-300"
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
            <Button
              className="w-full mt-6"
              disabled={!city}
              onClick={() => setStep(1)}
            >
              Continue
            </Button>
          </>
        )}

        {step === 1 && (
          <>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              When should we alert you?
            </h2>
            <p className="text-sm text-gray-500 mb-6">
              We&apos;ll send an email when AQI is forecast to reach this level within 24 hours.
            </p>
            <div className="space-y-2">
              {AQI_THRESHOLDS.map((t) => (
                <button
                  key={t}
                  onClick={() => setThreshold(t)}
                  className={`w-full py-3 px-4 rounded-xl border text-sm font-medium text-left transition-all ${
                    threshold === t
                      ? "border-teal-500 bg-teal-50 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300"
                      : "border-gray-200 dark:border-gray-700 hover:border-teal-300"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            <div className="flex gap-3 mt-6">
              <Button variant="secondary" onClick={() => setStep(0)}>Back</Button>
              <Button className="flex-1" onClick={() => setStep(2)}>Continue</Button>
            </div>
          </>
        )}

        {step === 2 && (
          <>
            <div className="text-center">
              <div className="w-16 h-16 bg-teal-100 dark:bg-teal-900/30 rounded-full flex items-center justify-center mx-auto mb-4 text-3xl">
                ok
              </div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">You&apos;re all set!</h2>
              <p className="text-sm text-gray-500 mb-6">
                Monitoring <strong>{city}</strong> for{" "}
                <strong>{threshold}</strong> or worse air quality.
                We&apos;ll email you when the AI forecast crosses this level.
              </p>
              <Button
                className="w-full"
                onClick={handleFinish}
                disabled={submitting}
              >
                {submitting ? "Setting up..." : "Go to Dashboard"}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
