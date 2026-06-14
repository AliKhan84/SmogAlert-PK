/**
 * Typed API client for SmogAlert PK backend.
 * All fetch calls include the Clerk session token from useAuth().
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

export interface CurrentAqi {
  city: string;
  aqi: number | null;
  category: string | null;
  pm25: number | null;
  pm10: number | null;
  no2: number | null;
  updated_at: string | null;
  source: string | null;
}

export interface ForecastPoint {
  hour: number;
  timestamp: string;
  pm25_predicted: number;
  lower_ci: number;
  upper_ci: number;
  aqi_category: string | null;
}

export interface ForecastResponse {
  city: string;
  generated_at: string;
  points: ForecastPoint[];
}

export interface AqiReading {
  city: string;
  timestamp: string;
  pm25: number | null;
  pm10: number | null;
  no2: number | null;
  so2: number | null;
  co: number | null;
  o3: number | null;
  aqi_calculated: number | null;
  aqi_category: string | null;
  is_anomaly: boolean;
  source: string | null;
}

export interface AlertHistory {
  id: number;
  city: string;
  aqi_level: string;
  aqi_value: number | null;
  message_en: string;
  channel: string;
  status: string;
  sent_at: string;
}

export interface UserProfile {
  id: number;
  email: string;
  name: string | null;
  phone_whatsapp: string | null;
  locations: { id: number; city: string; is_primary: boolean }[];
}

export interface AlertPreferences {
  channels: string;
  aqi_threshold: string;
  advance_hours: number;
  language: string;
}

async function apiFetch<T>(
  path: string,
  token: string | null,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

// --- Public endpoints (no auth required) ---

export const getCurrentAqi = () =>
  apiFetch<CurrentAqi[]>("/aqi/current", null);

export const getCityHistory = (city: string, limit = 168) =>
  apiFetch<AqiReading[]>(`/aqi/${city}/history?limit=${limit}`, null);

export const getCityForecast = (city: string) =>
  apiFetch<ForecastResponse>(`/aqi/${city}/forecast`, null);

// --- Authenticated endpoints ---

export const getProfile = (token: string) =>
  apiFetch<UserProfile>("/users/me", token);

export const updateProfile = (token: string, body: Partial<UserProfile>) =>
  apiFetch<UserProfile>("/users/me", token, { method: "PUT", body: JSON.stringify(body) });

export const getPreferences = (token: string) =>
  apiFetch<AlertPreferences>("/users/me/preferences", token);

export const updatePreferences = (token: string, body: Partial<AlertPreferences>) =>
  apiFetch<AlertPreferences>("/users/me/preferences", token, {
    method: "PUT",
    body: JSON.stringify(body),
  });

export const getAlertHistory = (token: string, limit = 50) =>
  apiFetch<AlertHistory[]>(`/alerts?limit=${limit}`, token);

export const sendTestAlert = (token: string) =>
  apiFetch<{ status: string }>("/alerts/test", token, { method: "POST" });

export const addLocation = (token: string, city: string, isPrimary = false) =>
  apiFetch("/users/me/locations", token, {
    method: "POST",
    body: JSON.stringify({ city, is_primary: isPrimary }),
  });

export const deleteLocation = (token: string, locationId: number) =>
  apiFetch(`/users/me/locations/${locationId}`, token, { method: "DELETE" });
