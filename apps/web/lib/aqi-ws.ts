/**
 * React hook for the Railway AQI WebSocket endpoint.
 * Replaces the Supabase Realtime subscription that pointed at empty Supabase tables.
 *
 * The server sends {"type":"aqi_update","data":[{city,aqi,category,pm25,updated_at},...]}
 * immediately on connect and every 5 minutes. It also broadcasts on each live scrape.
 *
 * Reconnects automatically with exponential back-off (1s → 2s → 4s → … → 30s cap).
 */
"use client";

import { useEffect, useRef } from "react";

// Convert the REST base URL to a WebSocket URL.
// http://localhost:8000/api  →  ws://localhost:8000/api/aqi/ws
// https://api.railway.app/api → wss://api.railway.app/api/aqi/ws
function getWsUrl(): string {
  const apiBase =
    process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";
  return apiBase.replace(/^http/, "ws") + "/aqi/ws";
}

export interface WsAqiItem {
  city: string;
  aqi: number | null;
  category: string | null;
  pm25: number | null;
  updated_at: string | null;
}

export function useAqiWebSocket(
  onUpdate: (data: WsAqiItem[]) => void
): void {
  // Keep a ref to the latest callback so the WS closure never goes stale.
  const callbackRef = useRef(onUpdate);
  useEffect(() => {
    callbackRef.current = onUpdate;
  });

  useEffect(() => {
    // Skip during SSR — WebSocket requires a browser environment.
    if (typeof window === "undefined") return;

    const url = getWsUrl();
    let ws: WebSocket | null = null;
    let retryMs = 1_000;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    function connect() {
      if (closed) return;
      ws = new WebSocket(url);

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data as string) as {
            type: string;
            data: WsAqiItem[];
          };
          if (msg.type === "aqi_update" && Array.isArray(msg.data)) {
            callbackRef.current(msg.data);
            retryMs = 1_000; // reset back-off on successful message
          }
        } catch {
          // Ignore malformed frames
        }
      };

      ws.onclose = () => {
        if (!closed) {
          retryTimer = setTimeout(connect, retryMs);
          retryMs = Math.min(retryMs * 2, 30_000);
        }
      };

      ws.onerror = () => {
        // onclose fires after onerror — reconnect handled there
        ws?.close();
      };
    }

    connect();

    return () => {
      closed = true;
      if (retryTimer !== null) clearTimeout(retryTimer);
      ws?.close();
    };
  }, []); // intentionally empty — connect once, keep alive via reconnect logic
}
