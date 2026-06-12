"use client";

import { useEffect, useRef } from "react";
import { CITY_COORDS, aqiCategoryColor } from "@/lib/aqi-utils";
import type { CurrentAqi } from "@/lib/api";

interface PakistanMapProps {
  readings: CurrentAqi[];
  selectedCity: string;
  onCitySelect: (city: string) => void;
}

export function PakistanMap({ readings, selectedCity, onCitySelect }: PakistanMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === "undefined" || mapInstanceRef.current) return;

    // Dynamic import to avoid SSR issues with Leaflet
    import("leaflet").then((L) => {
      if (!mapRef.current || mapInstanceRef.current) return;

      // Fix default icon path broken by webpack
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      });

      const map = L.map(mapRef.current!).setView([30.3753, 69.3451], 5);
      mapInstanceRef.current = map;

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
      }).addTo(map);

      readings.forEach((r) => {
        const coords = CITY_COORDS[r.city];
        if (!coords) return;
        const color = aqiCategoryColor(r.category);
        const icon = L.divIcon({
          html: `<div style="background:${color};color:#000;border:2px solid white;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:11px;box-shadow:0 2px 6px rgba(0,0,0,0.3)">${r.aqi != null ? Math.round(r.aqi) : "?"}</div>`,
          className: "",
          iconSize: [40, 40],
        });
        const marker = L.marker(coords, { icon }).addTo(map);
        marker.bindPopup(
          `<b>${r.city}</b><br/>AQI: ${r.aqi != null ? Math.round(r.aqi) : "N/A"}<br/>${r.category ?? ""}`
        );
        marker.on("click", () => onCitySelect(r.city));
      });
    });

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [readings, onCitySelect]);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 overflow-hidden shadow-sm">
      <div ref={mapRef} style={{ height: 320, width: "100%" }} />
    </div>
  );
}
