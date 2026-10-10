// src/usage.ts
// Usage tracking stored in localStorage under local date keys: "usage_YYYY-MM-DD"

export interface DailyUsage {
  requests: number;
  words: number;
  audioSeconds: number;
}

export interface UsageLimits {
  dailyRequests: number;
  dailyWords: number;
}

export const DEFAULT_USAGE_LIMITS: UsageLimits = {
  dailyRequests: 1500,
  dailyWords: 100000,
};

// Strips bracket tags like [excited], [calm] before word counting
export function stripBracketTags(text: string): string {
  return text.replace(/\[[^\]]*\]/g, "");
}

// Counts words after removing bracket tags
export function countWords(text: string): number {
  const cleaned = stripBracketTags(text).trim();
  if (!cleaned) return 0;
  return cleaned.split(/\s+/).filter(Boolean).length;
}

// Returns YYYY-MM-DD in local time
export function getTodayKey(): string {
  const d = new Date();
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `usage_${year}-${month}-${day}`;
}

export function getToday(): DailyUsage {
  const key = getTodayKey();
  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return { requests: 0, words: 0, audioSeconds: 0 };
    }
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return { requests: 0, words: 0, audioSeconds: 0 };
    }
    const parsed = JSON.parse(raw);
    return {
      requests: typeof parsed.requests === "number" && !isNaN(parsed.requests) ? parsed.requests : 0,
      words: typeof parsed.words === "number" && !isNaN(parsed.words) ? parsed.words : 0,
      audioSeconds: typeof parsed.audioSeconds === "number" && !isNaN(parsed.audioSeconds) ? parsed.audioSeconds : 0,
    };
  } catch (err) {
    console.warn("Failed to read today's usage from localStorage:", err);
    return { requests: 0, words: 0, audioSeconds: 0 };
  }
}

function saveToday(usage: DailyUsage): void {
  const key = getTodayKey();
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.setItem(key, JSON.stringify(usage));
    }
  } catch (err) {
    console.warn("Failed to save today's usage to localStorage:", err);
  }
}

export function addRequest(): DailyUsage {
  const current = getToday();
  const updated: DailyUsage = {
    ...current,
    requests: current.requests + 1,
  };
  saveToday(updated);
  return updated;
}

export function addWords(n: number): DailyUsage {
  const current = getToday();
  const updated: DailyUsage = {
    ...current,
    words: current.words + Math.max(0, Math.round(n)),
  };
  saveToday(updated);
  return updated;
}

export function addAudioSeconds(s: number): DailyUsage {
  const current = getToday();
  const updated: DailyUsage = {
    ...current,
    audioSeconds: Number((current.audioSeconds + Math.max(0, s)).toFixed(2)),
  };
  saveToday(updated);
  return updated;
}

// Mark requests as exhausted (set to limit) upon 429
export function setRequestsExhausted(limit: number): DailyUsage {
  const current = getToday();
  const updated: DailyUsage = {
    ...current,
    requests: Math.max(current.requests, limit),
  };
  saveToday(updated);
  return updated;
}

export function getUsageLimits(): UsageLimits {
  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return DEFAULT_USAGE_LIMITS;
    }
    const raw = window.localStorage.getItem("usage_limits");
    if (!raw) return DEFAULT_USAGE_LIMITS;
    const parsed = JSON.parse(raw);
    const dailyRequests =
      typeof parsed.dailyRequests === "number" && parsed.dailyRequests > 0
        ? parsed.dailyRequests
        : DEFAULT_USAGE_LIMITS.dailyRequests;
    const dailyWords =
      typeof parsed.dailyWords === "number" && parsed.dailyWords > 0
        ? parsed.dailyWords
        : DEFAULT_USAGE_LIMITS.dailyWords;
    return { dailyRequests, dailyWords };
  } catch {
    return DEFAULT_USAGE_LIMITS;
  }
}

export function saveUsageLimits(limits: UsageLimits): void {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.setItem("usage_limits", JSON.stringify(limits));
    }
  } catch (err) {
    console.warn("Failed to save usage limits to localStorage:", err);
  }
}

// Persian numeral formatter for digits when language is 'fa'
export function toPersianDigits(n: number | string): string {
  const str = String(n);
  const persianDigits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
  return str.replace(/\d/g, (d) => persianDigits[parseInt(d, 10)]);
}
