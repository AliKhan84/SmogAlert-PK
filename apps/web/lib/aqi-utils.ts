/** AQI colour scale and category utilities (US EPA standard). */

export const AQI_CATEGORIES = [
  "Good",
  "Moderate",
  "Unhealthy for Sensitive Groups",
  "Unhealthy",
  "Very Unhealthy",
  "Hazardous",
] as const;

export type AqiCategory = (typeof AQI_CATEGORIES)[number];

export const AQI_COLORS: Record<string, string> = {
  Good: "#00e400",
  Moderate: "#ffff00",
  "Unhealthy for Sensitive Groups": "#ff7e00",
  Unhealthy: "#ff0000",
  "Very Unhealthy": "#8f3f97",
  Hazardous: "#7e0023",
};

export const AQI_BG_CLASSES: Record<string, string> = {
  Good: "bg-green-100 text-green-800",
  Moderate: "bg-yellow-100 text-yellow-800",
  "Unhealthy for Sensitive Groups": "bg-orange-100 text-orange-800",
  Unhealthy: "bg-red-100 text-red-800",
  "Very Unhealthy": "bg-purple-100 text-purple-800",
  Hazardous: "bg-red-900 text-white",
};

export function pm25ToCategory(pm25: number): AqiCategory {
  if (pm25 < 12) return "Good";
  if (pm25 < 35.5) return "Moderate";
  if (pm25 < 55.5) return "Unhealthy for Sensitive Groups";
  if (pm25 < 150.5) return "Unhealthy";
  if (pm25 < 250.5) return "Very Unhealthy";
  return "Hazardous";
}

export function aqiCategoryColor(category: string | null): string {
  return AQI_COLORS[category ?? ""] ?? "#9ca3af";
}

export function aqiBadgeClass(category: string | null): string {
  return AQI_BG_CLASSES[category ?? ""] ?? "bg-gray-100 text-gray-800";
}

export const CITY_COORDS: Record<string, [number, number]> = {
  Islamabad: [33.6844, 73.0479],
  Karachi: [24.8607, 67.0011],
  Lahore: [31.5204, 74.3587],
  Peshawar: [34.0151, 71.5249],
  Quetta: [30.1798, 66.975],
};

export const PAKISTAN_CITIES = Object.keys(CITY_COORDS);
