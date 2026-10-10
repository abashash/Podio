export const DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts";

// Clip model: always 30 seconds, MP3, 48 kHz stereo. Preview model; names and behavior may change.
export const MUSIC_MODEL = "lyria-3-clip-preview";
export const MUSIC_CLIP_SECONDS = 30;

export interface ModelOption {
  value: string;
  label: string;
}

export const KNOWN_MUSIC_MODELS: ModelOption[] = [
  { value: "lyria-3-clip-preview", label: "lyria-3-clip-preview (default)" },
];

export const KNOWN_TTS_MODELS: ModelOption[] = [
  { value: DEFAULT_TTS_MODEL, label: `${DEFAULT_TTS_MODEL} (default)` },
  { value: "gemini-2.5-pro-preview-tts", label: "gemini-2.5-pro-preview-tts" },
  { value: "gemini-2.5-flash-preview-tts", label: "gemini-2.5-flash-preview-tts" },
  { value: "custom", label: "Custom model..." },
];

export const VOICES_BY_MODEL: Record<string, string[]> = {
  [DEFAULT_TTS_MODEL]: ["Rami", "Leda", "Puck", "Charon", "Kore", "Fenrir", "Aoede"],
  "gemini-2.5-pro-preview-tts": ["Rami", "Leda", "Puck", "Charon", "Kore", "Fenrir", "Aoede"],
  "gemini-2.5-flash-preview-tts": ["Rami", "Leda", "Puck", "Charon", "Kore", "Fenrir", "Aoede"],
};

export const DEFAULT_VOICES: Record<string, { voice1: string; voice2: string }> = {
  [DEFAULT_TTS_MODEL]: { voice1: "Rami", voice2: "Leda" },
  "gemini-2.5-pro-preview-tts": { voice1: "Rami", voice2: "Leda" },
  "gemini-2.5-flash-preview-tts": { voice1: "Rami", voice2: "Leda" },
};

export interface SpeakerNames {
  role1: string;
  role2: string;
}

export const DEFAULT_SPEAKER_NAMES: SpeakerNames = {
  role1: "Arash",
  role2: "Sara",
};

export function getSpeakerNames(): SpeakerNames {
  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return DEFAULT_SPEAKER_NAMES;
    }
    const raw = localStorage.getItem("speaker_names");
    if (!raw) return DEFAULT_SPEAKER_NAMES;
    const parsed = JSON.parse(raw);
    return {
      role1: typeof parsed.role1 === "string" && parsed.role1.trim() ? parsed.role1.trim() : DEFAULT_SPEAKER_NAMES.role1,
      role2: typeof parsed.role2 === "string" && parsed.role2.trim() ? parsed.role2.trim() : DEFAULT_SPEAKER_NAMES.role2,
    };
  } catch {
    return DEFAULT_SPEAKER_NAMES;
  }
}

export function saveSpeakerNames(names: SpeakerNames): void {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      localStorage.setItem("speaker_names", JSON.stringify(names));
    }
  } catch (err) {
    console.warn("Failed to save speaker_names to localStorage:", err);
  }
}
