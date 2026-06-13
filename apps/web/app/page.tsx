import Link from "next/link";
import { AqiGauge } from "@/components/dashboard/AqiGauge";
import { NavAuth } from "@/components/NavAuth";
import type { CurrentAqi } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

async function getAqiData(): Promise<CurrentAqi[]> {
  try {
    const res = await fetch(`${API_BASE}/aqi/current`, {
      next: { revalidate: 300 }, // ISR: revalidate every 5 minutes
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export default async function LandingPage() {
  const aqiData = await getAqiData();

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-white dark:from-gray-950 dark:to-gray-900">
      {/* Nav */}
      <nav className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur border-b border-gray-100 dark:border-gray-800">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <span className="font-bold text-teal-700 dark:text-teal-400 text-lg">
            SmogAlert PK
          </span>
          <NavAuth />
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-4 pt-20 pb-16 text-center">
        <div className="inline-block bg-teal-100 text-teal-800 text-xs font-medium px-3 py-1 rounded-full mb-4">
          Covering 5 Pakistani cities
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white leading-tight mb-4">
          Get air quality alerts{" "}
          <span className="text-teal-600">before it&apos;s dangerous</span>
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-8">
          SmogAlert PK uses AI to forecast PM2.5 levels 24 hours ahead and sends
          you bilingual alerts in English and Urdu before smog hits your city.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            href="/sign-up"
            className="bg-teal-600 text-white px-8 py-3 rounded-xl font-medium hover:bg-teal-700 transition-colors"
          >
            Start free — no credit card
          </Link>
          <Link
            href="/dashboard"
            className="bg-white border border-gray-200 text-gray-700 px-8 py-3 rounded-xl font-medium hover:bg-gray-50 transition-colors dark:bg-gray-800 dark:border-gray-700 dark:text-gray-200"
          >
            View live AQI map
          </Link>
        </div>
      </section>

      {/* Live AQI cards */}
      {aqiData.length > 0 && (
        <section className="max-w-6xl mx-auto px-4 pb-16">
          <h2 className="text-center text-sm font-medium text-gray-500 uppercase tracking-wider mb-6">
            Live Air Quality Index
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            {aqiData.map((r) => (
              <AqiGauge
                key={r.city}
                city={r.city}
                aqi={r.aqi}
                category={r.category}
                updatedAt={r.updated_at}
              />
            ))}
          </div>
        </section>
      )}

      {/* How it works */}
      <section className="bg-gray-50 dark:bg-gray-800/50 py-16">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-2xl font-bold text-center text-gray-900 dark:text-white mb-10">
            How it works
          </h2>
          <div className="grid sm:grid-cols-3 gap-8">
            {[
              {
                step: "1",
                title: "We collect live data",
                desc: "Hourly readings from WAQI, OpenAQ, and government EPA sensors across Islamabad, Lahore, Karachi, Peshawar, and Quetta.",
              },
              {
                step: "2",
                title: "AI forecasts 24 hours ahead",
                desc: "Our Prophet model predicts PM2.5 levels for the next 24 hours with confidence intervals — before conditions deteriorate.",
              },
              {
                step: "3",
                title: "You get alerted early",
                desc: "We send bilingual (English + Urdu) email alerts when AQI is forecast to cross your chosen threshold.",
              },
            ].map(({ step, title, desc }) => (
              <div key={step} className="text-center">
                <div className="w-10 h-10 bg-teal-600 text-white rounded-full flex items-center justify-center font-bold text-lg mx-auto mb-3">
                  {step}
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section className="max-w-4xl mx-auto px-4 py-16">
        <h2 className="text-2xl font-bold text-center text-gray-900 dark:text-white mb-10">
          Simple pricing
        </h2>
        <div className="grid sm:grid-cols-2 gap-6 max-w-2xl mx-auto">
          {[
            {
              name: "Free",
              price: "$0",
              features: ["1 city", "Email alerts", "Up to 3 alerts/day", "24-hour forecast"],
              cta: "Get started free",
              highlight: false,
            },
            {
              name: "Pro",
              price: "$3/month",
              features: [
                "All 5 cities",
                "Email + WhatsApp + SMS",
                "Unlimited alerts",
                "Anomaly source reports",
              ],
              cta: "Upgrade to Pro",
              highlight: true,
            },
          ].map(({ name, price, features, cta, highlight }) => (
            <div
              key={name}
              className={`rounded-2xl border p-6 ${highlight ? "border-teal-500 bg-teal-50 dark:bg-teal-900/20" : "border-gray-200 dark:border-gray-700"}`}
            >
              <h3 className="font-bold text-xl mb-1 text-gray-900 dark:text-white">{name}</h3>
              <p className="text-3xl font-bold text-teal-600 mb-4">{price}</p>
              <ul className="space-y-2 mb-6">
                {features.map((f) => (
                  <li key={f} className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                    <span className="text-teal-500">checkmark</span> {f}
                  </li>
                ))}
              </ul>
              <Link
                href="/sign-up"
                className={`block text-center py-2 rounded-lg text-sm font-medium transition-colors ${highlight ? "bg-teal-600 text-white hover:bg-teal-700" : "bg-gray-100 text-gray-800 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200"}`}
              >
                {cta}
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-100 dark:border-gray-800 py-8">
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-4 text-sm text-gray-500">
          <span>2026 SmogAlert PK</span>
          <div className="flex gap-6">
            <Link href="/dashboard" className="hover:text-gray-900 dark:hover:text-white">Dashboard</Link>
            <Link href="/settings" className="hover:text-gray-900 dark:hover:text-white">Settings</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
