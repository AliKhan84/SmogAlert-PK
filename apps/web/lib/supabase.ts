/**
 * Supabase browser client — used for Realtime subscriptions only.
 * Auth is handled by Clerk; we use the anon key here for public read access.
 */
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

// Guard: createClient throws if supabaseUrl is empty (happens during SSR without env vars).
// All callers should check `supabase !== null` before use.
export const supabase =
  supabaseUrl && supabaseAnonKey
    ? createClient(supabaseUrl, supabaseAnonKey)
    : null;
