import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronUp,
  Clipboard,
  Copy,
  Download,
  ExternalLink,
  FileCode,
  FileText,
  Gauge,
  Globe,
  Layers,
  Loader2,
  Mic2,
  Moon,
  Music,
  Play,
  Plus,
  Radio,
  RotateCcw,
  Sliders,
  Sparkles,
  Sun,
  Trash2,
  Users,
  Volume2,
} from "lucide-react";
import { GoogleGenAI } from "@google/genai";
import {
  AudioOutputConfig,
  BlockTiming,
  DEFAULT_AUDIO_OUTPUT,
  MusicAnchor,
  MusicClip,
  MusicPlacement,
  base64ToUint8Array,
  computeBlockTimings,
  computePlacementStartTime,
  concatUint8Arrays,
  createWavBlob,
  extractPcmPayload,
  getAudioOutputConfig,
  mixPodcastWithMusicPlacements,
  parseAudioMimeType,
  processPodcastAudioOutput,
  saveAudioOutputConfig,
  validatePlacement,
} from "./audioUtils";
import {
  ParsedDirectivesResult,
  parseScriptDirectives,
  resolvePlacementSpan,
} from "./directives";
import {
  DEFAULT_SPEAKER_NAMES,
  DEFAULT_TTS_MODEL,
  DEFAULT_VOICES,
  KNOWN_MUSIC_MODELS,
  KNOWN_TTS_MODELS,
  MUSIC_CLIP_SECONDS,
  MUSIC_MODEL,
  SpeakerNames,
  VOICES_BY_MODEL,
  getSpeakerNames,
  saveSpeakerNames,
} from "./config";
import {
  DailyUsage,
  UsageLimits,
  addAudioSeconds,
  addRequest,
  addWords,
  countWords,
  getToday,
  getTodayKey,
  getUsageLimits,
  saveUsageLimits,
  setRequestsExhausted,
  toPersianDigits,
} from "./usage";
import { createTranslator } from "./i18n";

import enScriptRaw from "../script.en.txt?raw";
import faScriptRaw from "../script.fa.txt?raw";
import arScriptRaw from "../script.ar.txt?raw";
import pythonScriptRaw from "../make_podcast.py?raw";

export type SupportedLang = "en" | "fa" | "ar";

export interface PodcastBlock {
  speaker: "Rami" | "Leda";
  style: string;
  text: string;
  lineNum?: number;
}

export interface ValidationError {
  lineNum: number;
  problem: string;
  preview: string;
}

interface ChunkLog {
  chunkIndex: number;
  firstBlock: number;
  lastBlock: number;
  pcmBytes: number;
  attempts: number;
}

interface SynthesisResult {
  fileName: string;
  url: string;
  durationSeconds: number;
  chunkLogs: ChunkLog[];
  format?: "wav" | "mp3";
  sampleRate?: number;
  channels?: "mono" | "stereo";
  mp3Bitrate?: number;
}

// Synthesis state type: "idle" | "in_progress" | "done" | "error"
interface SynthesisStatus {
  state: "idle" | "in_progress" | "done" | "error";
  chunksDone: number;
  chunksTotal: number;
  message: string;
}

const RAW_SCRIPTS_BY_LANG: Record<SupportedLang, string> = {
  en: enScriptRaw,
  fa: faScriptRaw,
  ar: arScriptRaw,
};

// Fixed configuration
const CHUNK_SIZE = 20;
const MAX_RETRIES = 3;
const RETRY_WAIT_MS = 5000;

const SPEAKER_MAP: Record<string, string> = {
  Rami: "Speaker 1",
  Leda: "Speaker 2",
};

// Client-side parser following exact rules
export function parseScriptInput(rawContent: string): {
  blocks: PodcastBlock[];
  errors: ValidationError[];
} {
  const lines = rawContent.split(/\r?\n/);
  const blocks: PodcastBlock[] = [];
  const errors: ValidationError[] = [];

  const hasStartMarker = lines.some((l) => l.trim() === "=== SCRIPT START ===");
  const hasEndMarker = lines.some((l) => l.trim() === "=== SCRIPT END ===");
  const useMarkers = hasStartMarker && hasEndMarker;

  let insideMarkers = false;

  for (let idx = 0; idx < lines.length; idx++) {
    const lineNum = idx + 1;
    const rawLine = lines[idx];
    const stripped = rawLine.trim();

    if (useMarkers) {
      if (stripped === "=== SCRIPT START ===") {
        insideMarkers = true;
        continue;
      }
      if (stripped === "=== SCRIPT END ===") {
        insideMarkers = false;
        break;
      }
      if (!insideMarkers) {
        continue;
      }
    }

    if (!stripped) {
      continue;
    }

    // Ignore comments starting with #
    if (stripped.startsWith("#")) {
      continue;
    }

    const firstBar = rawLine.indexOf("|");
    const secondBar = rawLine.indexOf("|", firstBar + 1);

    if (firstBar === -1 || secondBar === -1) {
      errors.push({
        lineNum,
        problem: "Line does not have three parts separated by '|'",
        preview: rawLine.slice(0, 60),
      });
      continue;
    }

    const speaker = rawLine.slice(0, firstBar).trim();
    const style = rawLine.slice(firstBar + 1, secondBar).trim();
    const text = rawLine.slice(secondBar + 1).trim();

    if (speaker !== "Rami" && speaker !== "Leda") {
      errors.push({
        lineNum,
        problem: `Speaker must be exactly 'Rami' or 'Leda' (got '${speaker}')`,
        preview: rawLine.slice(0, 60),
      });
      continue;
    }

    blocks.push({
      speaker: speaker as "Rami" | "Leda",
      style,
      text,
      lineNum,
    });
  }

  return { blocks, errors };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"script" | "cli" | "spec">("script");
  const [selectedLang, setSelectedLang] = useState<SupportedLang>("en");
  const [scriptRawText, setScriptRawText] = useState<string>(RAW_SCRIPTS_BY_LANG.en);
  const [originalRawText, setOriginalRawText] = useState<string>(RAW_SCRIPTS_BY_LANG.en);
  const [requestMode, setRequestMode] = useState<boolean>(false);
  const [writingScript, setWritingScript] = useState<boolean>(false);
  const [scriptGenerationError, setScriptGenerationError] = useState<string | null>(null);
  const [pythonScript] = useState<string>(pythonScriptRaw);

  // Model & Voice configuration state with localStorage persistence
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("tts_model");
      if (saved && saved.trim()) return saved.trim();
    }
    return "gemini-3.8-flash-tts";
  });
  const [customModelInput, setCustomModelInput] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("tts_model");
      const isKnown = KNOWN_TTS_MODELS.some((m) => m.value === saved);
      if (saved && !isKnown) return saved;
    }
    return "";
  });

  const effectiveModel = useMemo(() => {
    if (selectedModel === "custom") {
      return customModelInput.trim() || DEFAULT_TTS_MODEL;
    }
    return selectedModel || DEFAULT_TTS_MODEL;
  }, [selectedModel, customModelInput]);

  const defaultVoicesForModel = useMemo(() => {
    return DEFAULT_VOICES[effectiveModel] || { voice1: "Rami", voice2: "Leda" };
  }, [effectiveModel]);

  const [voice1, setVoice1] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("tts_voice1");
      if (saved && saved.trim()) return saved.trim();
    }
    return "Rami";
  });
  const [voice2, setVoice2] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("tts_voice2");
      if (saved && saved.trim()) return saved.trim();
    }
    return "Leda";
  });

  // When model changes, update voices to model defaults if custom voices not stored
  const handleModelChange = (newVal: string) => {
    setSelectedModel(newVal);
    if (newVal === "custom") {
      const target = customModelInput.trim() || DEFAULT_TTS_MODEL;
      localStorage.setItem("tts_model", target);
    } else {
      localStorage.setItem("tts_model", newVal);
      const defaults = DEFAULT_VOICES[newVal] || { voice1: "Rami", voice2: "Leda" };
      setVoice1(defaults.voice1);
      setVoice2(defaults.voice2);
      localStorage.setItem("tts_voice1", defaults.voice1);
      localStorage.setItem("tts_voice2", defaults.voice2);
    }
  };

  const handleCustomModelChange = (val: string) => {
    setCustomModelInput(val);
    const target = val.trim() || DEFAULT_TTS_MODEL;
    localStorage.setItem("tts_model", target);
  };

  const handleVoice1Change = (v: string) => {
    setVoice1(v);
    localStorage.setItem("tts_voice1", v);
  };

  const handleVoice2Change = (v: string) => {
    setVoice2(v);
    localStorage.setItem("tts_voice2", v);
  };

  const availableVoices = useMemo(() => {
    return (
      VOICES_BY_MODEL[effectiveModel] || [
        "Rami",
        "Leda",
        "Puck",
        "Charon",
        "Kore",
        "Fenrir",
        "Aoede",
      ]
    );
  }, [effectiveModel]);

  // Usage tracking state
  const [todayUsage, setTodayUsage] = useState<DailyUsage>(() => getToday());
  const [usageLimits, setUsageLimits] = useState<UsageLimits>(() => getUsageLimits());
  const [sessionRequests, setSessionRequests] = useState<number>(0);

  // Voice preview state
  const [previewLoading1, setPreviewLoading1] = useState<boolean>(false);
  const [previewLoading2, setPreviewLoading2] = useState<boolean>(false);
  const [previewError1, setPreviewError1] = useState<string | null>(null);
  const [previewError2, setPreviewError2] = useState<string | null>(null);
  const [previewAudioUrl1, setPreviewAudioUrl1] = useState<string | null>(null);
  const [previewAudioUrl2, setPreviewAudioUrl2] = useState<string | null>(null);
  const previewUrl1Ref = useRef<string | null>(null);
  const previewUrl2Ref = useRef<string | null>(null);

  // Cleanup preview object URLs on unmount
  useEffect(() => {
    return () => {
      if (previewUrl1Ref.current) URL.revokeObjectURL(previewUrl1Ref.current);
      if (previewUrl2Ref.current) URL.revokeObjectURL(previewUrl2Ref.current);
    };
  }, []);

  const isQuotaExhaustedError = (err: unknown): boolean => {
    if (!err) return false;
    const anyErr = err as any;
    if (anyErr?.status === 429 || anyErr?.code === 429 || anyErr?.error?.code === 429) return true;
    if (anyErr?.status === "RESOURCE_EXHAUSTED" || anyErr?.code === "RESOURCE_EXHAUSTED") return true;
    const msg = (anyErr?.message || String(err)).toUpperCase();
    return msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED") || msg.includes("QUOTA EXCEEDED");
  };

  const handleTestVoice = async (role: 1 | 2, voiceName: string) => {
    const apiKey =
      (typeof process !== "undefined" && process.env?.GEMINI_API_KEY) ||
      (typeof import.meta !== "undefined" &&
        (import.meta as any).env?.VITE_GEMINI_API_KEY) ||
      "";

    if (!apiKey) {
      const msg = t("GEMINI_API_KEY is not defined in the environment.");
      if (role === 1) setPreviewError1(msg);
      else setPreviewError2(msg);
      return;
    }

    if (role === 1) {
      setPreviewLoading1(true);
      setPreviewError1(null);
    } else {
      setPreviewLoading2(true);
      setPreviewError2(null);
    }

    try {
      // Real request: count before generateContent call
      addRequest();
      setTodayUsage(getToday());
      setSessionRequests((prev) => prev + 1);

      const ai = new GoogleGenAI({ apiKey });
      const response = await ai.models.generateContent({
        model: effectiveModel,
        contents: [
          {
            role: "user",
            parts: [{ text: "Hello, this is a short voice test." }],
          },
        ],
        config: {
          temperature: 1,
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName,
              },
            },
          },
        },
      });

      const extractedBuffers: Uint8Array[] = [];
      let detectedMime = "audio/L16;rate=24000";
      const candidateParts = response.candidates?.[0]?.content?.parts ?? [];
      for (const part of candidateParts) {
        if (part.inlineData?.data) {
          if (part.inlineData.mimeType) {
            detectedMime = part.inlineData.mimeType;
          }
          const rawBytes = base64ToUint8Array(part.inlineData.data);
          extractedBuffers.push(extractPcmPayload(rawBytes));
        }
      }

      if (extractedBuffers.length === 0) {
        throw new Error("Gemini TTS response did not contain audio bytes.");
      }

      const pcm = concatUint8Arrays(extractedBuffers);
      const wavBlob = createWavBlob(pcm, detectedMime);
      const url = URL.createObjectURL(wavBlob);

      if (role === 1) {
        if (previewUrl1Ref.current) URL.revokeObjectURL(previewUrl1Ref.current);
        previewUrl1Ref.current = url;
        setPreviewAudioUrl1(url);
      } else {
        if (previewUrl2Ref.current) URL.revokeObjectURL(previewUrl2Ref.current);
        previewUrl2Ref.current = url;
        setPreviewAudioUrl2(url);
      }
    } catch (err) {
      if (isQuotaExhaustedError(err)) {
        setRequestsExhausted(usageLimits.dailyRequests);
        setTodayUsage(getToday());
        const quotaMsg = t(
          t("Quota exceeded for this key today. Try again after the quota resets or enable billing.")
        );
        if (role === 1) setPreviewError1(quotaMsg);
        else setPreviewError2(quotaMsg);
      } else {
        const errMsg = err instanceof Error ? err.message : String(err);
        if (role === 1) setPreviewError1(errMsg);
        else setPreviewError2(errMsg);
      }
    } finally {
      if (role === 1) setPreviewLoading1(false);
      else setPreviewLoading2(false);
    }
  };

  // Background music state
  const [musicEnabled, setMusicEnabled] = useState<boolean>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("music_enabled") !== "false";
    }
    return true;
  });
  const [musicClips, setMusicClips] = useState<MusicClip[]>([]);
  const [generatedMusicPrompt, setGeneratedMusicPrompt] = useState<string | null>(null);
  const [musicPlacements, setMusicPlacements] = useState<MusicPlacement[]>([]);
  const [generatingMusic, setGeneratingMusic] = useState<boolean>(false);
  const [musicError, setMusicError] = useState<string | null>(null);
  const [selectedPreviewClipId, setSelectedPreviewClipId] = useState<string | null>(null);
  const musicAbortRef = useRef<AbortController | null>(null);

  const handleToggleMusic = (enabled: boolean) => {
    setMusicEnabled(enabled);
    if (typeof window !== "undefined") {
      localStorage.setItem("music_enabled", enabled ? "true" : "false");
    }
  };

  const [selectedMusicModel, setSelectedMusicModel] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("music_model");
      if (saved && saved.trim()) return saved.trim();
    }
    return MUSIC_MODEL;
  });

  const handleMusicModelChange = (model: string) => {
    setSelectedMusicModel(model);
    if (typeof window !== "undefined") {
      localStorage.setItem("music_model", model);
    }
  };

  // Speaker names state with localStorage persistence
  const [speakerNames, setSpeakerNames] = useState<SpeakerNames>(() => getSpeakerNames());

  const handleSpeakerNameChange = (role: "role1" | "role2", value: string) => {
    const updated = {
      ...speakerNames,
      [role]: value,
    };
    setSpeakerNames(updated);
    saveSpeakerNames({
      role1: updated.role1.trim() || DEFAULT_SPEAKER_NAMES.role1,
      role2: updated.role2.trim() || DEFAULT_SPEAKER_NAMES.role2,
    });
  };

  // Audio output configuration state with localStorage persistence
  const [audioOutput, setAudioOutput] = useState<AudioOutputConfig>(() => getAudioOutputConfig());

  const handleAudioOutputChange = (updates: Partial<AudioOutputConfig>) => {
    const updated = { ...audioOutput, ...updates };
    setAudioOutput(updated);
    saveAudioOutputConfig(updated);
  };

  const handleGenerateMusic = async (
    isAdditional = false,
    promptOverride?: string
  ): Promise<MusicClip | null> => {
    const effectivePrompt =
      promptOverride?.trim() || parsedDirectives.musicPrompt?.trim() || "";
    if (!effectivePrompt) return null;

    setGeneratingMusic(true);
    setMusicError(null);
    const controller = new AbortController();
    musicAbortRef.current = controller;

    try {
      // Real request: increment usage before generateContent/interactions call
      addRequest();
      setTodayUsage(getToday());
      setSessionRequests((prev) => prev + 1);

      const res = await fetch("/api/music/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept-Language": selectedLang,
        },
        body: JSON.stringify({ prompt: effectivePrompt }),
        signal: controller.signal,
      });

      if (!res.ok) {
        if (res.status === 429) {
          setRequestsExhausted(usageLimits.dailyRequests);
          setTodayUsage(getToday());
          throw new Error(
            t("Quota exceeded for this key today. Try again after the quota resets or enable billing.")
          );
        }
        const errJson = await res.json().catch(() => ({}));
        throw new Error(
          errJson.error?.message || errJson.error || `Failed to generate music (${res.status})`
        );
      }

      const data = await res.json();
      if (!data.audioBase64) {
        throw new Error(t("API response did not contain audio data."));
      }

      const binary = atob(data.audioBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: data.mimeType || "audio/mp3" });
      const audioUrl = URL.createObjectURL(blob);

      const nextClipNum = isAdditional ? musicClips.length + 1 : 1;
      const newClip: MusicClip = {
        id: `clip_${Date.now()}_${nextClipNum}`,
        clipNumber: nextClipNum,
        base64: data.audioBase64,
        audioUrl,
        durationSeconds: 30,
      };
      setGeneratedMusicPrompt(effectivePrompt);

      if (isAdditional) {
        setMusicClips((prev) => [...prev, newClip]);
        setSelectedPreviewClipId(newClip.id);
      } else {
        setMusicClips([newClip]);
        setSelectedPreviewClipId(newClip.id);
        if (musicPlacements.length === 0) {
          setMusicPlacements([
            {
              id: `p_${Date.now()}_start`,
              clipId: newClip.id,
              anchor: "start",
              anchorBlock: 1,
              clipFrom: 0,
              clipTo: 8,
              volumeDb: -18,
              fadeIn: 1.5,
              fadeOut: 1.5,
              duckSpeech: true,
            },
            {
              id: `p_${Date.now()}_end`,
              clipId: newClip.id,
              anchor: "end",
              anchorBlock: 1,
              clipFrom: 22,
              clipTo: 30,
              volumeDb: -18,
              fadeIn: 1.5,
              fadeOut: 1.5,
              duckSpeech: false,
            },
          ]);
        }
      }
      return newClip;
    } catch (err: any) {
      if (err.name === "AbortError") {
        setMusicError(t("Music generation cancelled."));
      } else if (isQuotaExhaustedError(err)) {
        setRequestsExhausted(usageLimits.dailyRequests);
        setTodayUsage(getToday());
        setMusicError(
          t("Quota exceeded for this key today. Try again after the quota resets or enable billing.")
        );
      } else {
        setMusicError(err.message || String(err));
      }
      return null;
    } finally {
      setGeneratingMusic(false);
      musicAbortRef.current = null;
    }
  };

  const handleCancelMusic = () => {
    if (musicAbortRef.current) {
      musicAbortRef.current.abort();
    }
  };

  const handleAddPlacement = () => {
    const defaultClipId = musicClips[0]?.id || "clip_1";
    const newPlacement: MusicPlacement = {
      id: `p_${Date.now()}_${musicPlacements.length + 1}`,
      clipId: defaultClipId,
      anchor: "start",
      anchorBlock: 1,
      clipFrom: 0,
      clipTo: 10,
      volumeDb: -18,
      fadeIn: 1.5,
      fadeOut: 1.5,
      duckSpeech: true,
    };
    setMusicPlacements((prev) => [...prev, newPlacement]);
  };

  const handleRemovePlacement = (id: string) => {
    setMusicPlacements((prev) => prev.filter((p) => p.id !== id));
  };

  const handleUpdatePlacement = (id: string, updates: Partial<MusicPlacement>) => {
    setMusicPlacements((prev) =>
      prev.map((p) => (p.id === id ? { ...p, ...updates } : p))
    );
  };

  const [synthesizing, setSynthesizing] = useState<boolean>(false);
  const [synthError, setSynthError] = useState<string | null>(null);
  const [synthResult, setSynthResult] = useState<SynthesisResult | null>(null);
  const [progress, setProgress] = useState<SynthesisStatus | null>(null);

  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [parsedBlocks, setParsedBlocks] = useState<PodcastBlock[]>([]);
  const [parsedDirectives, setParsedDirectives] = useState<ParsedDirectivesResult>(() =>
    parseScriptDirectives(RAW_SCRIPTS_BY_LANG.en, 30)
  );
  const [copiedCli, setCopiedCli] = useState<boolean>(false);
  const [resetMessage, setResetMessage] = useState<string | null>(null);

  // Debounced parsing ref
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Parse initial script on load
  useEffect(() => {
    const { blocks, errors } = parseScriptInput(RAW_SCRIPTS_BY_LANG.en);
    setParsedBlocks(blocks);
    const dirs = parseScriptDirectives(RAW_SCRIPTS_BY_LANG.en, blocks.length);
    setParsedDirectives(dirs);
    const allErrors: ValidationError[] = [
      ...errors,
      ...dirs.errors.map((e) => ({
        lineNum: e.lineNum,
        problem: e.problem,
        preview: e.preview,
      })),
    ];
    setValidationErrors(allErrors);
  }, []);

  // Detect unsaved changes compared to default loaded text for selected language
  const hasUnsavedChanges = useMemo(() => {
    return scriptRawText !== originalRawText;
  }, [scriptRawText, originalRawText]);

  // Handle language switch
  const handleLanguageChange = (nextLang: SupportedLang) => {
    if (nextLang === selectedLang) return;
    if (hasUnsavedChanges) {
      const confirmed = window.confirm(
        t("You have unsaved changes in the current script. Switching languages will discard unsaved edits. Continue?")
      );
      if (!confirmed) {
        return;
      }
    }
    setSelectedLang(nextLang);
    const content = RAW_SCRIPTS_BY_LANG[nextLang] ?? "";
    setScriptRawText(content);
    setOriginalRawText(content);

    const { blocks, errors } = parseScriptInput(content);
    setParsedBlocks(blocks);
    const dirs = parseScriptDirectives(content, blocks.length);
    setParsedDirectives(dirs);
    const allErrors: ValidationError[] = [
      ...errors,
      ...dirs.errors.map((e) => ({
        lineNum: e.lineNum,
        problem: e.problem,
        preview: e.preview,
      })),
    ];
    setValidationErrors(allErrors);
    setSynthError(null);
  };

  // Debounced live validation on typing
  const handleTextareaChange = (value: string) => {
    setScriptRawText(value);

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      const { blocks, errors } = parseScriptInput(value);
      setParsedBlocks(blocks);
      const dirs = parseScriptDirectives(value, blocks.length);
      setParsedDirectives(dirs);
      const allErrors: ValidationError[] = [
        ...errors,
        ...dirs.errors.map((e) => ({
          lineNum: e.lineNum,
          problem: e.problem,
          preview: e.preview,
        })),
      ];
      setValidationErrors(allErrors);
    }, 300);
  };

  const handleGenerateScript = async () => {
    const request = scriptRawText.trim();
    if (!request) {
      setScriptGenerationError(t("A short request is required."));
      return;
    }

    setWritingScript(true);
    setScriptGenerationError(null);
    try {
      const response = await fetch("/api/script/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept-Language": selectedLang,
        },
        body: JSON.stringify({ request, lang: selectedLang }),
      });
      const result = (await response.json()) as {
        script?: unknown;
        error?: unknown;
      };
      if (!response.ok) {
        throw new Error(
          typeof result.error === "string"
            ? result.error
            : t("Could not draft a valid script. Check the request and try again.")
        );
      }
      if (typeof result.script !== "string") {
        throw new Error(t("Could not draft a valid script. Check the request and try again."));
      }

      const { blocks, errors } = parseScriptInput(result.script);
      const directives = parseScriptDirectives(result.script, blocks.length);
      if (errors.length > 0 || directives.errors.length > 0 || blocks.length === 0) {
        throw new Error(t("Could not draft a valid script. Check the request and try again."));
      }

      handleTextareaChange(result.script);
      setParsedBlocks(blocks);
      setParsedDirectives(directives);
      setValidationErrors([]);
      setRequestMode(false);
    } catch (error) {
      setScriptGenerationError(error instanceof Error ? error.message : String(error));
    } finally {
      setWritingScript(false);
    }
  };

  // Reset to original file content
  const handleReset = () => {
    const content = RAW_SCRIPTS_BY_LANG[selectedLang] ?? "";
    setScriptRawText(content);
    setOriginalRawText(content);
    const { blocks, errors } = parseScriptInput(content);
    setParsedBlocks(blocks);
    const dirs = parseScriptDirectives(content, blocks.length);
    setParsedDirectives(dirs);
    const allErrors: ValidationError[] = [
      ...errors,
      ...dirs.errors.map((e) => ({
        lineNum: e.lineNum,
        problem: e.problem,
        preview: e.preview,
      })),
    ];
    setValidationErrors(allErrors);
    setResetMessage(t("Reset to script.{lang}.txt", { lang: selectedLang }));
    setTimeout(() => setResetMessage(null), 2500);
  };

  // Paste text from clipboard into editor
  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        handleTextareaChange(text);
      }
    } catch {
      // Clipboard read fallback if browser permissions deny direct read
    }
  };

  // Clear text editor
  const handleClear = () => {
    handleTextareaChange("");
  };

  // Synthesis executed directly in browser with @google/genai SDK
  const handleSynthesize = async () => {
    const { blocks, errors } = parseScriptInput(scriptRawText);
    if (errors.length > 0) {
      setSynthError(t("Please fix validation errors before synthesizing."));
      return;
    }
    if (blocks.length === 0) {
      setSynthError(t("No dialogue blocks found to synthesize."));
      return;
    }

    const apiKey =
      (typeof process !== "undefined" && process.env?.GEMINI_API_KEY) ||
      (typeof import.meta !== "undefined" &&
        (import.meta as any).env?.VITE_GEMINI_API_KEY) ||
      "";

    if (!apiKey) {
      setSynthError(
        t("GEMINI_API_KEY is not defined in the environment. Please ensure GEMINI_API_KEY is configured.")
      );
      return;
    }

    setSynthesizing(true);
    setSynthError(null);
    setSynthResult(null);

    const totalBlocks = blocks.length;
    const totalChunks = Math.ceil(totalBlocks / CHUNK_SIZE);
    const synthesisDirectives = parseScriptDirectives(scriptRawText, totalBlocks);

    setProgress({
      state: "in_progress",
      chunksDone: 0,
      chunksTotal: totalChunks,
      message: t("Starting synthesis ({done} of {total} chunks)", {
        done: 0,
        total: totalChunks,
      }),
    });

    const ai = new GoogleGenAI({ apiKey });
    const pcmChunks: Uint8Array[] = [];
    const chunkLogs: ChunkLog[] = [];
    let detectedMimeType = "audio/L16;rate=24000";

    const directives = parseScriptDirectives(scriptRawText, blocks.length);
    if (directives.errors.length > 0) {
      setSynthError(`Directive error on line ${directives.errors[0].lineNum}: ${directives.errors[0].problem}`);
      setSynthesizing(false);
      return;
    }

    const musicPromptFromScript = directives.musicPrompt?.trim();
    let currentMusicClips: MusicClip[] = [...musicClips];

    try {
      // If #MUSIC_PROMPT is specified in the box, send it to lyria-3-clip-preview once
      if (musicPromptFromScript) {
        setProgress({
          state: "in_progress",
          chunksDone: 0,
          chunksTotal: totalChunks,
          message: "Generating background music (Lyria 3)...",
        });

        addRequest();
        setTodayUsage(getToday());
        setSessionRequests((prev) => prev + 1);

        const musicRes = await fetch("/api/music/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: musicPromptFromScript }),
        });

        if (!musicRes.ok) {
          if (musicRes.status === 429) {
            setRequestsExhausted(usageLimits.dailyRequests);
            setTodayUsage(getToday());
            throw new Error(
              "Quota exceeded for this key today. Try again after the quota resets or enable billing."
            );
          }
          const errJson = await musicRes.json().catch(() => ({}));
          throw new Error(
            errJson.error?.message || errJson.error || `Failed to generate music (${musicRes.status})`
          );
        }

        const data = await musicRes.json();
        if (!data.audioBase64) {
          throw new Error("Lyria music API did not return audio data.");
        }

        const binary = atob(data.audioBase64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], { type: data.mimeType || "audio/mp3" });
        const audioUrl = URL.createObjectURL(blob);

        const newClip: MusicClip = {
          id: `clip_${Date.now()}_1`,
          clipNumber: 1,
          base64: data.audioBase64,
          audioUrl,
          durationSeconds: 30,
        };
        currentMusicClips = [newClip];
        setMusicClips([newClip]);
        setMusicEnabled(true);
      }
      for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        const startIdx = chunkIdx * CHUNK_SIZE;
        const endIdx = Math.min(startIdx + CHUNK_SIZE, totalBlocks);
        const chunkBlocks = blocks.slice(startIdx, endIdx);
        const firstBlockNum = startIdx + 1;
        const lastBlockNum = endIdx;

        // Never send style label to API; keep for UI and console logs only
        const parts = chunkBlocks.map((block, idx) => {
          const rawSpeaker = block.speaker?.trim();
          const mappedSpeaker = SPEAKER_MAP[rawSpeaker] ?? "Speaker 1";
          console.log(
            `[Chunk ${chunkIdx + 1}/${totalChunks}] Block ${startIdx + idx + 1}: Speaker=${rawSpeaker} (${mappedSpeaker}) | Style=${block.style}`
          );
          return {
            text: String(block.text ?? ""),
            speechMetadata: {
              speaker: mappedSpeaker,
            },
          };
        });

        let chunkPcm: Uint8Array | null = null;
        let lastError: unknown = null;
        let usedAttempts = 0;

        for (let attempt = 1; attempt <= MAX_RETRIES + 1; attempt++) {
          usedAttempts = attempt;
          try {
            // Count request before each generateContent call, including retries
            addRequest();
            setTodayUsage(getToday());
            setSessionRequests((prev) => prev + 1);

            const response = await ai.models.generateContent({
              model: effectiveModel,
              contents: [
                {
                  role: "user",
                  parts,
                },
              ],
              config: {
                temperature: 1,
                responseModalities: ["AUDIO"],
                speechConfig: {
                  multiSpeakerVoiceConfig: {
                    speakerVoiceConfigs: [
                      {
                        speaker: "Speaker 1",
                        voiceConfig: {
                          prebuiltVoiceConfig: { voiceName: voice1 },
                        },
                      },
                      {
                        speaker: "Speaker 2",
                        voiceConfig: {
                          prebuiltVoiceConfig: { voiceName: voice2 },
                        },
                      },
                    ],
                  },
                },
              },
            });

            const extractedBuffers: Uint8Array[] = [];
            const candidateParts = response.candidates?.[0]?.content?.parts ?? [];
            for (const part of candidateParts) {
              if (part.inlineData?.data) {
                if (part.inlineData.mimeType) {
                  detectedMimeType = part.inlineData.mimeType;
                }
                const rawBytes = base64ToUint8Array(part.inlineData.data);
                extractedBuffers.push(extractPcmPayload(rawBytes));
              }
            }

            if (extractedBuffers.length === 0) {
              throw new Error("Gemini TTS response did not contain audio bytes.");
            }

            chunkPcm = concatUint8Arrays(extractedBuffers);
            break;
          } catch (err) {
            lastError = err;
            if (isQuotaExhaustedError(err)) {
              setRequestsExhausted(usageLimits.dailyRequests);
              setTodayUsage(getToday());
              const quotaMsg =
                t("Quota exceeded for this key today. Try again after the quota resets or enable billing.");
              setProgress({
                state: "error",
                chunksDone: chunkIdx,
                chunksTotal: totalChunks,
                message: quotaMsg,
              });
              setSynthError(quotaMsg);
              return;
            }

            if (attempt <= MAX_RETRIES) {
              console.warn(
                `Chunk ${chunkIdx + 1}/${totalChunks} (blocks ${firstBlockNum}-${lastBlockNum}) failed attempt ${attempt}/${MAX_RETRIES}. Retrying in ${RETRY_WAIT_MS}ms...`
              );
              await sleep(RETRY_WAIT_MS);
            } else {
              const errMsg = err instanceof Error ? err.message : String(err);
              setProgress({
                state: "error",
                chunksDone: chunkIdx,
                chunksTotal: totalChunks,
                message: t("Chunk {chunk} failed after {retries} retries.", {
                  chunk: `${chunkIdx + 1}/${totalChunks}`,
                  retries: MAX_RETRIES,
                }),
              });
              throw new Error(
                `Chunk ${chunkIdx + 1}/${totalChunks} covering blocks ${firstBlockNum} to ${lastBlockNum} failed after ${MAX_RETRIES} retries: ${errMsg}`
              );
            }
          }
        }

        if (chunkPcm) {
          pcmChunks.push(chunkPcm);
          const chunkWords = chunkBlocks.reduce(
            (acc, b) => acc + countWords(b.text || ""),
            0
          );
          const { bitsPerSample, rate } = parseAudioMimeType(detectedMimeType);
          const chunkSeconds = Number(
            (chunkPcm.length / (rate * (bitsPerSample / 8))).toFixed(2)
          );
          addWords(chunkWords);
          addAudioSeconds(chunkSeconds);
          setTodayUsage(getToday());

          chunkLogs.push({
            chunkIndex: chunkIdx + 1,
            firstBlock: firstBlockNum,
            lastBlock: lastBlockNum,
            pcmBytes: chunkPcm.length,
            attempts: usedAttempts,
          });

          setProgress({
            state: "in_progress",
            chunksDone: chunkIdx + 1,
            chunksTotal: totalChunks,
            message: t("Completed chunk {done} of {total}", {
              done: chunkIdx + 1,
              total: totalChunks,
            }),
          });
        } else if (lastError) {
          throw lastError;
        }
      }

      // Combine PCM and generate playable WAV blob in browser
      const combinedPcm = concatUint8Arrays(pcmChunks);
      const { bitsPerSample, rate } = parseAudioMimeType(detectedMimeType);
      const baseDurationSeconds = Number(
        (combinedPcm.length / (rate * (bitsPerSample / 8))).toFixed(2)
      );
      const timestamp = Date.now();

      let synthesisMusicClips = musicClips;
      const scriptMusicPrompt = synthesisDirectives.musicPrompt?.trim() || "";
      if (
        musicEnabled &&
        scriptMusicPrompt &&
        (synthesisMusicClips.length === 0 || generatedMusicPrompt !== scriptMusicPrompt)
      ) {
        const generatedClip = await handleGenerateMusic(false, scriptMusicPrompt);
        if (!generatedClip) {
          throw new Error(t("Music generation failed. Review the music error and try again."));
        }
        synthesisMusicClips = [generatedClip];
      }

      // Check if background music is enabled and has clips and placements
      const effectivePlacements: MusicPlacement[] =
        synthesisDirectives.musicDirectives.length > 0 && synthesisMusicClips.length > 0
          ? synthesisDirectives.musicDirectives.map((d, dIdx) => ({
              id: `dir_p_${dIdx + 1}`,
              clipId: synthesisMusicClips[0]?.id || "clip_1",
              anchor:
                d.anchorType === "before"
                  ? "before_block"
                  : d.anchorType === "after"
                  ? "after_block"
                  : d.anchorType,
              anchorBlock: d.anchorBlock || 1,
              untilBlock: d.untilBlock,
              clipFrom: d.clipFrom,
              clipTo: d.clipTo,
              volumeDb: d.volumeDb,
              fadeIn: d.fadeIn,
              fadeOut: d.fadeOut,
              duckSpeech: d.duckSpeech,
            }))
          : musicPlacements.length > 0
          ? musicPlacements
          : synthesisMusicClips.length > 0 && scriptMusicPrompt
          ? [
              {
                id: `default_p_${synthesisMusicClips[0].id}_start`,
                clipId: synthesisMusicClips[0].id,
                anchor: "start",
                anchorBlock: 1,
                clipFrom: 0,
                clipTo: 8,
                volumeDb: -18,
                fadeIn: 1.5,
                fadeOut: 1.5,
                duckSpeech: true,
              },
              {
                id: `default_p_${synthesisMusicClips[0].id}_end`,
                clipId: synthesisMusicClips[0].id,
                anchor: "end",
                anchorBlock: 1,
                clipFrom: 22,
                clipTo: 30,
                volumeDb: -18,
                fadeIn: 1.5,
                fadeOut: 1.5,
                duckSpeech: false,
              },
            ]
          : [];

      const shouldMixMusic =
        musicEnabled &&
        Boolean(scriptMusicPrompt) &&
        synthesisMusicClips.length > 0 &&
        (effectivePlacements.length > 0 || (audioOutput.musicLeadIn > 0 || audioOutput.musicTail > 0));

      if (!shouldMixMusic) {
        let finalBlob: Blob;
        let finalExt: "wav" | "mp3" = audioOutput.format;
        let finalDuration = baseDurationSeconds;

        if (audioOutput.format === "mp3") {
          const res = await processPodcastAudioOutput(combinedPcm, detectedMimeType, audioOutput);
          finalBlob = res.blob;
          finalExt = "mp3";
          finalDuration = res.durationSeconds;
        } else if (audioOutput.sampleRate !== 24000 || audioOutput.channels === "stereo") {
          const res = await processPodcastAudioOutput(combinedPcm, detectedMimeType, audioOutput);
          finalBlob = res.blob;
          finalExt = "wav";
          finalDuration = res.durationSeconds;
        } else {
          // Unchanged byte-for-byte original path
          finalBlob = createWavBlob(combinedPcm, detectedMimeType);
          finalExt = "wav";
          finalDuration = baseDurationSeconds;
        }

        const audioUrl = URL.createObjectURL(finalBlob);
        const fileName = `podcast_${selectedLang}_${timestamp}.${finalExt}`;

        setProgress({
          state: "done",
          chunksDone: totalChunks,
          chunksTotal: totalChunks,
          message: `${t("Synthesis complete")}! ${fileName}`,
        });

        setSynthResult({
          fileName,
          url: audioUrl,
          durationSeconds: finalDuration,
          chunkLogs,
          format: finalExt,
          sampleRate: audioOutput.sampleRate,
          channels: audioOutput.channels,
          mp3Bitrate: audioOutput.mp3Bitrate,
        });
      } else {
        // Background music mix path
        setProgress({
          state: "in_progress",
          chunksDone: totalChunks,
          chunksTotal: totalChunks,
          message: t("Mixing background music placements..."),
        });

        const blockTimings = computeBlockTimings(blocks, chunkLogs, rate);
        const mixed = await mixPodcastWithMusicPlacements({
          speechPcm: combinedPcm,
          clips: synthesisMusicClips,
          placements: effectivePlacements,
          blockTimings,
          musicModes: synthesisDirectives.musicModeDirectives.map((m) => ({
            startBlock: m.startBlock,
            endBlock: m.endBlock,
            mode: m.mode,
          })),
          leadInSeconds: audioOutput.musicLeadIn,
          tailSeconds: audioOutput.musicTail,
          outputFormat: audioOutput.format,
          channels: audioOutput.channels,
          mp3Bitrate: audioOutput.mp3Bitrate,
        });

        const audioUrl = URL.createObjectURL(mixed.blob);
        const fileName = `podcast_${selectedLang}_${timestamp}.${mixed.extension}`;

        setProgress({
          state: "done",
          chunksDone: totalChunks,
          chunksTotal: totalChunks,
          message: `${t("Synthesis & Music mix complete")}! ${fileName}`,
        });

        setSynthResult({
          fileName,
          url: audioUrl,
          durationSeconds: mixed.durationSeconds,
          chunkLogs,
          format: mixed.extension,
          sampleRate: audioOutput.sampleRate,
          channels: audioOutput.channels,
          mp3Bitrate: audioOutput.mp3Bitrate,
        });
      }
    } catch (err) {
      setSynthError(err instanceof Error ? err.message : t("Synthesis failed."));
      setProgress({
        state: "error",
        chunksDone: chunkLogs.length,
        chunksTotal: totalChunks,
        message: t("Synthesis failed."),
      });
    } finally {
      setSynthesizing(false);
    }
  };

  const handleCopyCli = async () => {
    await navigator.clipboard.writeText(pythonScript);
    setCopiedCli(true);
    setTimeout(() => setCopiedCli(false), 2000);
  };

  const totalBlocks = parsedBlocks.length;
  const ramiBlocks = parsedBlocks.filter((b) => b.speaker === "Rami").length;
  const ledaBlocks = parsedBlocks.filter((b) => b.speaker === "Leda").length;
  const chunksCount = Math.max(0, Math.ceil(totalBlocks / 20));

  const isRtl = selectedLang === "fa" || selectedLang === "ar";
  const t = useMemo(() => createTranslator(selectedLang), [selectedLang]);

  const [openModels, setOpenModels] = useState<boolean>(true);
  const [openVoices, setOpenVoices] = useState<boolean>(true);
  const [openOutput, setOpenOutput] = useState<boolean>(true);
  const [openMusic, setOpenMusic] = useState<boolean>(true);
  const [openCreatePodcast, setOpenCreatePodcast] = useState<boolean>(true);
  const [openAdvanced, setOpenAdvanced] = useState<boolean>(false);

  const [openToolsRef, setOpenToolsRef] = useState<boolean>(false);
  const [openUsage, setOpenUsage] = useState<boolean>(false);
  const [openCli, setOpenCli] = useState<boolean>(false);
  const [openSpecs, setOpenSpecs] = useState<boolean>(false);
  const [isDark, setIsDark] = useState<boolean>(true);

  return (
    <div
      lang={selectedLang}
      dir={isRtl ? "rtl" : "ltr"}
      className="min-h-screen bg-[#0a0a0b] text-[#f5f5f7] flex flex-col selection:bg-[#c4ff3d]/30 selection:text-[#c4ff3d]"
    >
      {/* 1. Header (one row, under 64 px) */}
      <header className="h-14 flex items-center justify-between px-4 sm:px-6 border-b border-white/[0.08] bg-[#0a0a0b]/90 backdrop-blur-md sticky top-0 z-30">
        {/* Start side: app logo and name */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-[#c4ff3d] flex items-center justify-center text-[#0a0a0b] shadow-[0_0_15px_rgba(196,255,61,0.3)] shrink-0">
            <Radio className="w-4 h-4 stroke-[2.5]" />
          </div>
          <span className="text-base sm:text-lg font-bold tracking-tight text-white whitespace-nowrap">
            Audioq
          </span>
        </div>

        {/* Center / end side: compact language switch (فارسی / English / العربية) */}
        <div className="flex items-center gap-1.5 p-1 bg-[#121214] border border-white/[0.08] rounded-full">
          {(
            [
              { id: "fa", label: "فارسی" },
              { id: "en", label: "English" },
              { id: "ar", label: "العربية" },
            ] as const
          ).map((langItem) => (
            <button
              key={langItem.id}
              type="button"
              onClick={() => handleLanguageChange(langItem.id)}
              className={`px-3 py-1 text-xs rounded-full transition-all whitespace-nowrap ${
                selectedLang === langItem.id
                  ? "bg-[#c4ff3d] text-[#0a0a0b] font-semibold shadow-sm"
                  : "text-[#a1a1aa] hover:text-white"
              }`}
            >
              {langItem.label}
            </button>
          ))}
        </div>

        {/* End side: theme toggle only */}
        <div className="flex items-center">
          <button
            type="button"
            onClick={() => setIsDark((prev) => !prev)}
            title={t("Switch theme")}
            className="p-2 rounded-xl bg-[#121214] hover:bg-white/[0.08] text-[#a1a1aa] hover:text-[#f5f5f7] border border-white/[0.08] transition-colors"
          >
            {isDark ? (
              <Sun className="w-4 h-4 text-[#c4ff3d]" />
            ) : (
              <Moon className="w-4 h-4 text-[#c4ff3d]" />
            )}
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-[1280px] w-full mx-auto px-4 sm:px-6 py-6 flex-1 flex flex-col gap-6">
        {/* 2. Hero (small, centered, above the workspace) */}
        <div className="text-center py-2 max-w-2xl mx-auto space-y-2">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
            {t("Dual-Speaker Podcast Audio Generator")}
          </h1>
          <p className="text-xs sm:text-sm text-[#a1a1aa]">
            {t("Natural conversational synthesis in English, Persian, and Arabic.")}
          </p>
        </div>

        {/* 3. Workspace Grid (2 equal columns on desktop: 1fr 1fr; 1 column < 1024px) */}
        <div className="podcast-workspace-grid">
          {/* Left Column (50% on desktop): Script Editor Card */}
          <div className="w-full min-w-0 bg-[#121214] border border-white/[0.08] rounded-[20px] p-5 sm:p-6 flex flex-col gap-4 shadow-xl">
            {/* Top row inside card: title "Script", language buttons, and live counters on the end side */}
            <div className="flex flex-wrap items-center justify-between pb-3.5 border-b border-white/[0.07] gap-3">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-[#c4ff3d]" />
                <span className="text-sm font-semibold text-white tracking-wide">
                  {t("Script")}
                </span>
                <span className="text-xs text-[#52525b] font-mono hidden sm:inline-block">
                  script.{selectedLang}.txt
                </span>
              </div>

              {/* Language buttons (فارسی / English / العربية) */}
              <div className="flex items-center gap-1 p-0.5 bg-[#18181b] border border-white/[0.08] rounded-full">
                {(
                  [
                    { id: "fa", label: "فارسی" },
                    { id: "en", label: "English" },
                    { id: "ar", label: "العربية" },
                  ] as const
                ).map((langItem) => (
                  <button
                    key={langItem.id}
                    type="button"
                    onClick={() => handleLanguageChange(langItem.id)}
                    className={`px-2.5 py-1 text-xs rounded-full transition-all whitespace-nowrap ${
                      selectedLang === langItem.id
                        ? "bg-[#c4ff3d] text-[#0a0a0b] font-semibold shadow-sm"
                        : "text-[#a1a1aa] hover:text-white"
                    }`}
                  >
                    {langItem.label}
                  </button>
                ))}
              </div>

              {/* Counts line: N lines · X Rami · Y Leda */}
              <div className="flex items-center gap-2 text-xs font-mono text-[#a1a1aa] flex-wrap justify-end">
                <span>
                  <strong className="text-white font-medium">{totalBlocks}</strong> {t("lines")}
                </span>
                <span className="text-[#52525b]">·</span>
                <span>
                  <strong className="text-[#c4ff3d] font-medium">{ramiBlocks}</strong> {t("Rami")}
                </span>
                <span className="text-[#52525b]">·</span>
                <span>
                  <strong className="text-[#c4ff3d]/80 font-medium">{ledaBlocks}</strong> {t("Leda")}
                </span>
                <span className="text-[#52525b]">·</span>
                <span>
                  <strong className="text-[#c4ff3d] font-medium">{chunksCount}</strong> {t("chunks")}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-xs text-[#a1a1aa] cursor-pointer">
                <input
                  type="checkbox"
                  checked={requestMode}
                  disabled={writingScript}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    setRequestMode(enabled);
                    setScriptGenerationError(null);
                    if (enabled) handleTextareaChange("");
                  }}
                  className="w-3.5 h-3.5 accent-[#c4ff3d]"
                />
                {t("Write script from a request")}
              </label>
              {requestMode && (
                <button
                  type="button"
                  onClick={handleGenerateScript}
                  disabled={writingScript || !scriptRawText.trim()}
                  className="px-3 py-2 text-xs font-semibold text-[#0a0a0b] bg-[#c4ff3d] rounded-lg hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center gap-2"
                >
                  {writingScript && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  {t(writingScript ? "Writing script..." : "Write script")}
                </button>
              )}
            </div>
            {requestMode && (
              <p className="text-xs text-[#71717a]">
                {t("Describe your podcast idea, length, and language.")}
              </p>
            )}
            {scriptGenerationError && (
              <p role="alert" className="text-xs text-red-300">
                {scriptGenerationError}
              </p>
            )}

            {/* Middle: large textarea (min 24 rows) */}
            <textarea
              id="script-textarea"
              rows={24}
              dir="auto"
              value={scriptRawText}
              onChange={(e) => handleTextareaChange(e.target.value)}
              placeholder={
                requestMode
                  ? t("Describe your podcast idea, length, and language.")
                  : "=== SCRIPT START ===&#10;Rami | Promote | [excited] Welcome to our podcast!&#10;Leda | Friendly | [warmly] Thanks for joining us today!&#10;=== SCRIPT END ==="
              }
              className={`w-full p-4 sm:p-5 text-xs sm:text-sm font-mono bg-[#18181b] border rounded-xl focus:outline-none focus:border-[#c4ff3d] leading-relaxed text-[#f5f5f7] placeholder:text-[#52525b] transition-all resize-y min-h-[480px] ${
                validationErrors.length > 0 ? "border-red-800" : "border-white/[0.08]"
              } ${isRtl ? "text-right" : "text-left"}`}
            />

            {/* Bottom row: small action buttons (paste, load file, clear) */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3.5 border-t border-white/[0.07]">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handlePaste}
                  className="px-3 py-1.5 text-xs font-medium text-[#f5f5f7] bg-[#18181b] hover:bg-white/[0.08] border border-white/[0.08] rounded-xl transition-colors flex items-center gap-1.5"
                >
                  <Clipboard className="w-3.5 h-3.5 text-[#a1a1aa]" />
                  {t("Paste")}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-3 py-1.5 text-xs font-medium text-[#f5f5f7] bg-[#18181b] hover:bg-white/[0.08] border border-white/[0.08] rounded-xl transition-colors flex items-center gap-1.5"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-[#a1a1aa]" />
                  {t("Load File")}
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-3 py-1.5 text-xs font-medium text-[#a1a1aa] hover:text-red-400 bg-[#18181b] hover:bg-white/[0.08] border border-white/[0.08] rounded-xl transition-colors flex items-center gap-1.5"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  {t("Clear")}
                </button>
              </div>

              <div className="flex items-center gap-3 text-xs font-mono">
                {resetMessage && (
                  <span className="text-[#c4ff3d]">{resetMessage}</span>
                )}
                {hasUnsavedChanges && (
                  <span className="text-amber-400 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                    {t("Modified")}
                  </span>
                )}
                <span className="text-[#52525b] text-[11px]">
                  {t("Dir:")} {t(isRtl ? "RTL" : "LTR")}
                </span>
              </div>
            </div>

            {/* Validation errors appear below textarea as a short list with line numbers */}
            {validationErrors.length > 0 && !requestMode && (
              <div className="mt-2 p-3.5 bg-red-950/40 border border-red-800/80 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-red-300">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <span>{t("Validation Errors")} ({validationErrors.length}):</span>
                </div>
                <ul className="space-y-1 text-xs font-mono text-red-200 divide-y divide-red-900/40 max-h-44 overflow-y-auto pr-1">
                  {validationErrors.map((err, i) => (
                    <li key={i} className="pt-1 flex flex-col sm:flex-row sm:gap-2">
                      <span className="text-red-400 font-semibold shrink-0">
                        {t("Line {line}:", { line: err.lineNum })}
                      </span>
                      <span className="text-red-300">{err.problem}</span>
                      <span className="text-[#a1a1aa] truncate max-w-md">
                        &quot;{err.preview}&quot;
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Parsed Music Directives & Timeline Display */}
            {(parsedDirectives.musicPrompt ||
              parsedDirectives.musicDirectives.length > 0 ||
              parsedDirectives.musicModeDirectives.length > 0) && (
              <div className="mt-3 p-4 bg-[#18181b] border border-white/[0.08] rounded-xl space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
                  <div className="flex items-center gap-2 text-xs font-semibold text-white">
                    <Music className="w-4 h-4 text-[#c4ff3d]" />
                    <span>{t("Parsed Music Directives")}</span>
                  </div>
                  <span className="text-[11px] font-mono text-[#a1a1aa]">
                    {parsedDirectives.musicDirectives.length} placement(s) · {parsedDirectives.musicModeDirectives.length} mode rule(s)
                  </span>
                </div>

                {/* Prompt */}
                {parsedDirectives.musicPrompt && (
                  <div className="p-2.5 bg-[#121214] border border-white/[0.06] rounded-lg text-xs space-y-1">
                    <span className="text-[10px] text-[#a1a1aa] uppercase tracking-wider font-mono font-semibold">
                      Music Prompt
                    </span>
                    <p className="text-white italic">&quot;{parsedDirectives.musicPrompt}&quot;</p>
                  </div>
                )}

                {/* Placements Summary */}
                {parsedDirectives.musicDirectives.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-white">{t("Placements:")}</div>
                    <div className="grid grid-cols-1 gap-2">
                      {parsedDirectives.musicDirectives.map((d, i) => {
                        const span = resolvePlacementSpan(d, totalBlocks);

                        // Check if placement covers only silent blocks
                        const coveredBlocks = parsedBlocks
                          .map((_, idx) => idx + 1)
                          .filter((b) => b >= span.startBlockSpan && b <= span.endBlockSpan);
                        const isAllSilent =
                          coveredBlocks.length > 0 &&
                          coveredBlocks.every((b) => {
                            const explicitMode = parsedDirectives.musicModeDirectives
                              .filter((m) => b >= m.startBlock && b <= m.endBlock)
                              .map((m) => m.mode);
                            return explicitMode.includes("silent");
                          });

                        return (
                          <div
                            key={i}
                            className="p-2.5 bg-[#121214] border border-white/[0.06] rounded-lg text-xs font-mono space-y-1"
                          >
                            <div className="flex flex-wrap items-center justify-between gap-1 text-white">
                              <span className="font-semibold text-[#c4ff3d]">
                                #{i + 1} Anchor: {d.anchorType}
                                {d.anchorBlock ? `:${d.anchorBlock}` : ""}
                                {d.untilBlock ? ` until:${d.untilBlock}` : ""}
                              </span>
                              <span className="text-[#a1a1aa] text-[11px]">
                                Clip: {d.clipFrom}-{d.clipTo}s · Vol: {d.volumeDb}dB · Fade: {d.fadeIn},{d.fadeOut}s · Duck: {d.duckSpeech ? "on" : "off"}
                              </span>
                            </div>
                            <div className="text-[#a1a1aa] text-[11px]">
                              Span: <strong className="text-white font-medium">{span.description}</strong> (blocks {span.startBlockSpan}–{span.endBlockSpan})
                            </div>
                            {isAllSilent && (
                              <div className="text-amber-400 text-[11px] flex items-center gap-1.5 pt-0.5">
                                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                                <span>{t("Warning: This placement covers only silent blocks; it will produce no audible music sound during this span.")}</span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Per-Block Timeline List */}
                <div className="space-y-2 pt-2 border-t border-white/[0.06]">
                  <div className="flex items-center justify-between text-xs font-semibold text-white">
                    <span>{t("Per-Block Music Timeline")}</span>
                    <span className="text-[11px] font-mono text-[#a1a1aa]">
                      Total: {totalBlocks} dialogue block(s)
                    </span>
                  </div>

                  <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 divide-y divide-white/[0.04]">
                    {parsedBlocks.map((b, idx) => {
                      const blockNum = idx + 1;
                      const preview30 = b.text.slice(0, 30);

                      // Determine effective mode for this block
                      let mode: "normal" | "duck" | "silent" = "normal";
                      const coveringModes = parsedDirectives.musicModeDirectives.filter(
                        (m) => blockNum >= m.startBlock && blockNum <= m.endBlock
                      );
                      if (coveringModes.some((m) => m.mode === "silent")) {
                        mode = "silent";
                      } else if (coveringModes.some((m) => m.mode === "duck")) {
                        mode = "duck";
                      }

                      // Check covering placements
                      const coveringPlacements = parsedDirectives.musicDirectives
                        .map((d, pIdx) => {
                          const span = resolvePlacementSpan(d, totalBlocks);
                          const covers = blockNum >= span.startBlockSpan && blockNum <= span.endBlockSpan;
                          return covers ? `#${pIdx + 1}` : null;
                        })
                        .filter(Boolean);

                      return (
                        <div
                          key={blockNum}
                          className="pt-1.5 flex items-center justify-between gap-2 text-xs font-mono"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-[#a1a1aa] w-7 shrink-0 text-right">
                              #{blockNum}
                            </span>
                            <span className="text-white truncate max-w-[200px] sm:max-w-xs">
                              &quot;{preview30}&quot;
                            </span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <span
                              className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold ${
                                mode === "silent"
                                  ? "bg-red-500/20 text-red-400"
                                  : mode === "duck"
                                  ? "bg-amber-500/20 text-amber-300"
                                  : "bg-white/[0.05] text-[#a1a1aa]"
                              }`}
                            >
                              {t(mode)}
                            </span>
                            <span className="text-[10px] text-[#71717a]">
                              {coveringPlacements.length > 0
                                ? t("Placements: {placements}", {
                                    placements: coveringPlacements.join(", "),
                                  })
                                : t("No music")}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Total Output Length */}
                <div className="p-2.5 bg-[#121214] border border-white/[0.06] rounded-lg flex flex-wrap items-center justify-between text-xs font-mono text-[#a1a1aa]">
                  <span>{t("Timeline duration preview:")}</span>
                  <span className="text-white font-medium">
                    Speech: ~{((parsedBlocks.reduce((acc, b) => acc + countWords(b.text), 0) / 150) * 60).toFixed(1)}s
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Right Column (50% on desktop): Stack of Collapsible Cards */}
          <div className="w-full min-w-0 flex flex-col gap-4">
            {/* Card 1: Models */}
            <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                onClick={() => setOpenModels(!openModels)}
              >
                <div className="flex items-center gap-2.5">
                  <Layers className="w-4 h-4 text-[#c4ff3d]" />
                  <span className="text-sm font-semibold text-white tracking-wide">{t("Models")}</span>
                </div>
                <button
                  type="button"
                  aria-expanded={openModels}
                  aria-controls="card-models-body"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenModels(!openModels);
                  }}
                  className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                  title={t(openModels ? "Collapse" : "Expand")}
                >
                  <ChevronDown
                    className={`w-4 h-4 transform transition-transform duration-200 ${openModels ? "rotate-180" : ""}`}
                  />
                </button>
              </div>

              {openModels && (
                <div id="card-models-body" className="p-4 sm:p-5 pt-0 space-y-4 border-t border-white/[0.06]">
                  {/* TTS Model Selector Checkbox List */}
                  <div className="p-3.5 bg-[#18181b] border border-white/[0.06] rounded-xl space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-white">{t("TTS Model")}</span>
                      <span className="text-[11px] text-[#a1a1aa] font-mono">
                        {effectiveModel}
                      </span>
                    </div>

                    <div className="space-y-1.5">
                      {KNOWN_TTS_MODELS.map((opt) => {
                        const isChecked = selectedModel === opt.value;
                        return (
                          <label
                            key={opt.value}
                            onClick={() => handleModelChange(opt.value)}
                            className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all ${
                              isChecked
                                ? "bg-[#c4ff3d]/10 border-[#c4ff3d]/40 text-white"
                                : "bg-[#121214] border-white/[0.06] text-[#a1a1aa] hover:border-white/[0.15] hover:text-white"
                            }`}
                          >
                            <div className="flex items-center gap-2.5">
                              <input
                                type="checkbox"
                                checked={isChecked}
                                onChange={() => handleModelChange(opt.value)}
                                className="w-3.5 h-3.5 rounded border-white/20 text-[#c4ff3d] accent-[#c4ff3d] focus:ring-0 cursor-pointer"
                              />
                              <span className="text-xs font-mono">{opt.label}</span>
                            </div>
                            {isChecked && (
                              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#c4ff3d]/20 text-[#c4ff3d]">
                                {t("Active")}
                              </span>
                            )}
                          </label>
                        );
                      })}
                    </div>

                    {selectedModel === "custom" && (
                      <div className="space-y-1 pt-1 border-t border-white/[0.05]">
                        <label className="text-[11px] text-[#a1a1aa]">{t("Custom Model Name:")}</label>
                        <input
                          type="text"
                          value={customModelInput}
                          onChange={(e) => handleCustomModelChange(e.target.value)}
                          placeholder={`e.g. ${DEFAULT_TTS_MODEL}`}
                          className="w-full bg-[#121214] border border-white/[0.1] rounded-lg px-3 py-2 text-xs text-white placeholder-[#52525b] focus:outline-none focus:border-[#c4ff3d] font-mono"
                        />
                      </div>
                    )}

                    <p className="text-[11px] text-[#a1a1aa] leading-relaxed pt-1 border-t border-white/[0.05]">
                      Voices: Rami (Speaker 1), Leda (Speaker 2). Some models may not support multi-speaker mode; if synthesis fails with an unsupported-model error, choose the default model.
                    </p>
                  </div>

                  {/* Music Model Section */}
                  <div className="p-3.5 bg-[#18181b] border border-white/[0.06] rounded-xl space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Music className="w-3.5 h-3.5 text-[#c4ff3d]" />
                        <span className="text-xs font-semibold text-white">{t("Music model")}</span>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      {KNOWN_MUSIC_MODELS.map((mOpt) => {
                        const isSelected = selectedMusicModel === mOpt.value;
                        return (
                          <label
                            key={mOpt.value}
                            onClick={() => handleMusicModelChange(mOpt.value)}
                            className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all ${
                              isSelected
                                ? "bg-[#c4ff3d]/10 border-[#c4ff3d]/40 text-white"
                                : "bg-[#121214] border-white/[0.06] text-[#a1a1aa] hover:border-white/[0.15] hover:text-white"
                            }`}
                          >
                            <div className="flex items-center gap-2.5">
                              <input
                                type="radio"
                                name="music_model_radio"
                                checked={isSelected}
                                onChange={() => handleMusicModelChange(mOpt.value)}
                                className="w-3.5 h-3.5 text-[#c4ff3d] accent-[#c4ff3d] cursor-pointer"
                              />
                              <span className="text-xs font-mono">{mOpt.label}</span>
                            </div>
                            {isSelected && (
                              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#c4ff3d]/20 text-[#c4ff3d]">
                                {t("Active")}
                              </span>
                            )}
                          </label>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Card 2: Voices */}
            <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                onClick={() => setOpenVoices(!openVoices)}
              >
                <div className="flex items-center gap-2.5">
                  <Users className="w-4 h-4 text-[#c4ff3d]" />
                  <span className="text-sm font-semibold text-white tracking-wide">{t("Voices")}</span>
                </div>
                <button
                  type="button"
                  aria-expanded={openVoices}
                  aria-controls="card-voices-body"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenVoices(!openVoices);
                  }}
                  className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                  title={t(openVoices ? "Collapse" : "Expand")}
                >
                  <ChevronDown
                    className={`w-4 h-4 transform transition-transform duration-200 ${openVoices ? "rotate-180" : ""}`}
                  />
                </button>
              </div>

              {openVoices && (
                <div id="card-voices-body" className="p-4 sm:p-5 pt-0 space-y-3.5 border-t border-white/[0.06]">
                  {/* Speaker 1 (Role: Rami) */}
                  <div className="p-3 bg-[#18181b] border border-white/[0.06] rounded-xl flex flex-col gap-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0 flex-1">
                        <div className="w-6 h-6 rounded-lg bg-[#c4ff3d]/10 text-[#c4ff3d] flex items-center justify-center text-xs font-bold font-mono shrink-0">
                          1
                        </div>
                        <div className="min-w-0 flex-1">
                          <input
                            type="text"
                            value={speakerNames.role1}
                            onChange={(e) => handleSpeakerNameChange("role1", e.target.value)}
                            placeholder={DEFAULT_SPEAKER_NAMES.role1}
                            className="w-full bg-[#121214] border border-white/[0.1] rounded px-2 py-0.5 text-xs text-white font-medium focus:outline-none focus:border-[#c4ff3d]"
                          />
                          <div className="text-[11px] text-[#a1a1aa] font-mono mt-0.5 truncate">
                            Speaker 1 · Role: Rami
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <select
                          value={voice1}
                          onChange={(e) => handleVoice1Change(e.target.value)}
                          className="bg-[#121214] border border-white/[0.1] rounded-lg px-2 py-1 text-xs text-[#c4ff3d] font-mono focus:outline-none focus:border-[#c4ff3d]"
                        >
                          {availableVoices.map((v) => (
                            <option key={v} value={v}>
                              {v}
                            </option>
                          ))}
                        </select>
                        <button
                          type="button"
                          onClick={() => handleTestVoice(1, voice1)}
                          disabled={previewLoading1}
                          aria-label={t("Test voice")}
                          title={t("Test voice")}
                          className="p-1.5 bg-[#121214] hover:bg-white/[0.08] border border-white/[0.1] rounded-lg text-[#c4ff3d] transition-colors disabled:opacity-50"
                        >
                          {previewLoading1 ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Play className="w-3.5 h-3.5 fill-current" />
                          )}
                        </button>
                        <Mic2 className="w-3.5 h-3.5 text-[#c4ff3d]" />
                      </div>
                    </div>
                    {previewAudioUrl1 && (
                      <div className="flex items-center justify-end gap-2 pt-1 border-t border-white/[0.05]">
                        <span className="text-[10px] text-[#a1a1aa] font-mono">{t("Sample:")}</span>
                        <audio
                          controls
                          autoPlay
                          src={previewAudioUrl1}
                          className="h-6 w-44 rounded accent-[#c4ff3d]"
                        />
                      </div>
                    )}
                    {previewError1 && (
                      <div className="text-[11px] text-red-400 font-mono pt-1 border-t border-red-900/40">
                        {previewError1}
                      </div>
                    )}
                  </div>

                  {/* Speaker 2 (Role: Leda) */}
                  <div className="p-3 bg-[#18181b] border border-white/[0.06] rounded-xl flex flex-col gap-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0 flex-1">
                        <div className="w-6 h-6 rounded-lg bg-[#c4ff3d]/10 text-[#c4ff3d] flex items-center justify-center text-xs font-bold font-mono shrink-0">
                          2
                        </div>
                        <div className="min-w-0 flex-1">
                          <input
                            type="text"
                            value={speakerNames.role2}
                            onChange={(e) => handleSpeakerNameChange("role2", e.target.value)}
                            placeholder={DEFAULT_SPEAKER_NAMES.role2}
                            className="w-full bg-[#121214] border border-white/[0.1] rounded px-2 py-0.5 text-xs text-white font-medium focus:outline-none focus:border-[#c4ff3d]"
                          />
                          <div className="text-[11px] text-[#a1a1aa] font-mono mt-0.5 truncate">
                            Speaker 2 · Role: Leda
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <select
                          value={voice2}
                          onChange={(e) => handleVoice2Change(e.target.value)}
                          className="bg-[#121214] border border-white/[0.1] rounded-lg px-2 py-1 text-xs text-[#c4ff3d] font-mono focus:outline-none focus:border-[#c4ff3d]"
                        >
                          {availableVoices.map((v) => (
                            <option key={v} value={v}>
                              {v}
                            </option>
                          ))}
                        </select>
                        <button
                          type="button"
                          onClick={() => handleTestVoice(2, voice2)}
                          disabled={previewLoading2}
                          aria-label={t("Test voice")}
                          title={t("Test voice")}
                          className="p-1.5 bg-[#121214] hover:bg-white/[0.08] border border-white/[0.1] rounded-lg text-[#c4ff3d] transition-colors disabled:opacity-50"
                        >
                          {previewLoading2 ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Play className="w-3.5 h-3.5 fill-current" />
                          )}
                        </button>
                        <Mic2 className="w-3.5 h-3.5 text-[#c4ff3d]" />
                      </div>
                    </div>
                    {previewAudioUrl2 && (
                      <div className="flex items-center justify-end gap-2 pt-1 border-t border-white/[0.05]">
                        <span className="text-[10px] text-[#a1a1aa] font-mono">{t("Sample:")}</span>
                        <audio
                          controls
                          autoPlay
                          src={previewAudioUrl2}
                          className="h-6 w-44 rounded accent-[#c4ff3d]"
                        />
                      </div>
                    )}
                    {previewError2 && (
                      <div className="text-[11px] text-red-400 font-mono pt-1 border-t border-red-900/40">
                        {previewError2}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Card 3: Output */}
            <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                onClick={() => setOpenOutput(!openOutput)}
              >
                <div className="flex items-center gap-2.5">
                  <Sliders className="w-4 h-4 text-[#c4ff3d]" />
                  <span className="text-sm font-semibold text-white tracking-wide">{t("Output")}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono uppercase px-2 py-0.5 rounded bg-white/[0.05] text-[#c4ff3d]">
                    {audioOutput.format}
                  </span>
                  <button
                    type="button"
                    aria-expanded={openOutput}
                    aria-controls="card-output-body"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenOutput(!openOutput);
                    }}
                    className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                    title={t(openOutput ? "Collapse" : "Expand")}
                  >
                    <ChevronDown
                      className={`w-4 h-4 transform transition-transform duration-200 ${openOutput ? "rotate-180" : ""}`}
                    />
                  </button>
                </div>
              </div>

              {openOutput && (
                <div id="card-output-body" className="p-4 sm:p-5 pt-0 space-y-3.5 border-t border-white/[0.06]">
                  {/* Format: two radio buttons, WAV and MP3 */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-white block">{t("Format")}</label>
                    <div className="grid grid-cols-2 gap-2">
                      {(["wav", "mp3"] as const).map((fmt) => {
                        const isSelected = audioOutput.format === fmt;
                        return (
                          <label
                            key={fmt}
                            onClick={() => handleAudioOutputChange({ format: fmt })}
                            className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer transition-all ${
                              isSelected
                                ? "bg-[#c4ff3d]/10 border-[#c4ff3d]/40 text-white"
                                : "bg-[#18181b] border-white/[0.06] text-[#a1a1aa] hover:border-white/[0.15]"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <input
                                type="radio"
                                name="audio_format"
                                checked={isSelected}
                                onChange={() => handleAudioOutputChange({ format: fmt })}
                                className="w-3.5 h-3.5 text-[#c4ff3d] accent-[#c4ff3d] cursor-pointer"
                              />
                              <span className="text-xs font-mono uppercase font-semibold">{fmt}</span>
                            </div>
                            <span className="text-[10px] text-[#71717a] font-mono">
                              {t(fmt === "wav" ? "Lossless" : "Compressed")}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  {/* Sample rate */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <label className="font-semibold text-white">{t("Sample rate")}</label>
                      <span className="text-[11px] text-[#a1a1aa] font-mono">{audioOutput.sampleRate} Hz</span>
                    </div>
                    <select
                      value={audioOutput.sampleRate}
                      onChange={(e) =>
                        handleAudioOutputChange({
                          sampleRate: parseInt(e.target.value, 10) as 16000 | 24000 | 44100,
                        })
                      }
                      className="w-full bg-[#18181b] border border-white/[0.1] rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-[#c4ff3d]"
                    >
                      <option value={16000}>16,000 Hz</option>
                      <option value={24000}>24,000 Hz (default)</option>
                      <option value={44100}>44,100 Hz</option>
                    </select>
                  </div>

                  {/* Channels */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-white block">{t("Channels")}</label>
                    <div className="grid grid-cols-2 gap-2">
                      {(["mono", "stereo"] as const).map((ch) => {
                        const isSelected = audioOutput.channels === ch;
                        return (
                          <label
                            key={ch}
                            onClick={() => handleAudioOutputChange({ channels: ch })}
                            className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all ${
                              isSelected
                                ? "bg-[#c4ff3d]/10 border-[#c4ff3d]/40 text-white"
                                : "bg-[#18181b] border-white/[0.06] text-[#a1a1aa] hover:border-white/[0.15]"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <input
                                type="radio"
                                name="audio_channels"
                                checked={isSelected}
                                onChange={() => handleAudioOutputChange({ channels: ch })}
                                className="w-3.5 h-3.5 text-[#c4ff3d] accent-[#c4ff3d] cursor-pointer"
                              />
                              <span className="text-xs font-mono capitalize">{ch}</span>
                            </div>
                            <span className="text-[10px] text-[#71717a] font-mono">
                              {ch === "mono" ? "1-ch" : "2-ch"}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  {/* MP3 bitrate */}
                  <div className={`space-y-1.5 transition-opacity ${audioOutput.format === "wav" ? "opacity-40" : ""}`}>
                    <div className="flex justify-between items-center text-xs">
                      <label className="font-semibold text-white">{t("MP3 bitrate")}</label>
                      <span className="text-[11px] text-[#a1a1aa] font-mono">
                        {audioOutput.format === "wav" ? "N/A (WAV)" : `${audioOutput.mp3Bitrate} kbps`}
                      </span>
                    </div>
                    <select
                      value={audioOutput.mp3Bitrate}
                      disabled={audioOutput.format === "wav"}
                      onChange={(e) =>
                        handleAudioOutputChange({
                          mp3Bitrate: parseInt(e.target.value, 10) as 64 | 96 | 128 | 192,
                        })
                      }
                      className="w-full bg-[#18181b] border border-white/[0.1] rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-[#c4ff3d] disabled:cursor-not-allowed"
                    >
                      <option value={64}>64 kbps</option>
                      <option value={96}>96 kbps</option>
                      <option value={128}>128 kbps (default)</option>
                      <option value={192}>192 kbps</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Card 4: Music */}
            <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                onClick={() => setOpenMusic(!openMusic)}
              >
                <div className="flex items-center gap-2.5">
                  <Music className="w-4 h-4 text-[#c4ff3d]" />
                  <span className="text-sm font-semibold text-white tracking-wide">{t("Music")}</span>
                </div>
                <button
                  type="button"
                  aria-expanded={openMusic}
                  aria-controls="card-music-body"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenMusic(!openMusic);
                  }}
                  className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                  title={t(openMusic ? "Collapse" : "Expand")}
                >
                  <ChevronDown
                    className={`w-4 h-4 transform transition-transform duration-200 ${openMusic ? "rotate-180" : ""}`}
                  />
                </button>
              </div>

              {openMusic && (
                <div id="card-music-body" className="p-4 sm:p-5 pt-0 space-y-4 border-t border-white/[0.06]">
                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={musicEnabled}
                        onChange={(e) => handleToggleMusic(e.target.checked)}
                        className="w-3.5 h-3.5 rounded border-white/20 text-[#c4ff3d] accent-[#c4ff3d] focus:ring-0 cursor-pointer"
                      />
                      <span className="text-xs text-white font-medium">{t("Add background music")}</span>
                    </label>
                  </div>

                  {musicEnabled && (
                    <div className="space-y-3 pt-2 border-t border-white/[0.06]">
                      {/* Prompt is read from the #MUSIC_PROMPT directive in the script editor. */}
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-[11px] text-[#a1a1aa]">
                          <label>{t("Music prompt from script:")}</label>
                          <span className="font-mono">
                            {parsedDirectives.musicPrompt?.length || 0} chars
                          </span>
                        </div>
                        <p className="text-xs text-[#a1a1aa] leading-relaxed">
                          {parsedDirectives.musicPrompt ||
                            t("Add a #MUSIC_PROMPT: line to the script box.")}
                        </p>
                      </div>

                      {/* Generate / cancel music button */}
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => handleGenerateMusic(false)}
                          disabled={generatingMusic || !parsedDirectives.musicPrompt?.trim()}
                          className="flex-1 py-2 px-3 text-xs font-semibold text-[#0a0a0b] bg-[#c4ff3d] rounded-lg hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-1.5"
                        >
                          {generatingMusic ? (
                            <>
                              <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              <span>{t("Generating clip...")}</span>
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-3.5 h-3.5" />
                              <span>{t(musicClips.length > 0 ? "Regenerate music" : "Create music")}</span>
                            </>
                          )}
                        </button>
                        {generatingMusic && (
                          <button
                            type="button"
                            onClick={handleCancelMusic}
                            className="py-2 px-3 text-xs font-medium text-white bg-[#18181b] hover:bg-white/[0.08] border border-white/[0.1] rounded-lg transition-colors"
                          >
                            Cancel
                          </button>
                        )}
                      </div>
                        <p className="text-[11px] text-[#71717a] pt-1 leading-relaxed">
                          {t("When you press Create podcast, the app generates music from this prompt and places it according to your #MUSIC lines.")}
                        </p>

                      {musicError && (
                        <div className="p-2 bg-red-950/40 border border-red-900/60 rounded-lg text-[11px] text-red-300 font-mono">
                          {musicError}
                        </div>
                      )}

                      {/* Generated Clips */}
                      {musicClips.length > 0 && (
                        <div className="space-y-2 pt-1 border-t border-white/[0.05]">
                          <div className="flex items-center justify-between text-xs font-semibold text-white">
                            <span>{t("Generated Clips")} ({musicClips.length})</span>
                            <button
                              type="button"
                              onClick={() => handleGenerateMusic(true)}
                              disabled={generatingMusic || !parsedDirectives.musicPrompt?.trim()}
                              className="text-[11px] text-[#c4ff3d] hover:underline flex items-center gap-1"
                            >
                              <Plus className="w-3 h-3" />
                              {t("Generate another clip")}
                            </button>
                          </div>
                          {musicClips.map((clip) => (
                            <div key={clip.id} className="p-2 bg-[#121214] border border-white/[0.06] rounded-lg space-y-1.5">
                              <div className="flex items-center justify-between text-[11px] font-mono text-[#a1a1aa]">
                                <span className="text-white font-medium">
                                  {t("Clip #{number} ({duration})", {
                                    number: clip.clipNumber,
                                    duration: t("30s"),
                                  })}
                                </span>
                              </div>
                              <audio controls src={clip.audioUrl} className="w-full h-7 rounded accent-[#c4ff3d]" />
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Placements Section */}
                      <div className="space-y-2.5 pt-1 border-t border-white/[0.05]">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-white">{t("Placements")}</span>
                          <button
                            type="button"
                            onClick={handleAddPlacement}
                            className="px-2 py-1 text-[11px] font-medium text-[#0a0a0b] bg-[#c4ff3d] rounded hover:brightness-110 flex items-center gap-1"
                          >
                            <Plus className="w-3 h-3" />
                            {t("Add placement")}
                          </button>
                        </div>

                        {musicPlacements.length === 0 && (
                          <p className="text-[11px] text-[#71717a] italic">{t("No placements added yet.")}</p>
                        )}

                        {musicPlacements.map((p, idx) => (
                          <div key={p.id} className="p-2.5 bg-[#121214] border border-white/[0.06] rounded-lg space-y-2 text-xs">
                            <div className="flex items-center justify-between">
                              <span className="font-semibold text-white text-[11px]">Placement #{idx + 1}</span>
                              <button
                                type="button"
                                onClick={() => handleRemovePlacement(p.id)}
                                className="text-red-400 hover:text-red-300 p-1"
                                title={t("Remove placement")}
                              >
                                <Trash2 className="w-3 h-3" />
                              </button>
                            </div>

                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="text-[10px] text-[#a1a1aa] block">{t("Anchor")}</label>
                                <select
                                  value={p.anchor}
                                  onChange={(e) =>
                                    handleUpdatePlacement(p.id, { anchor: e.target.value as MusicAnchor })
                                  }
                                  className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                >
                                  <option value="start">{t("Start of podcast")}</option>
                                  <option value="end">{t("End of podcast")}</option>
                                  <option value="before_block">{t("Before block N")}</option>
                                  <option value="after_block">{t("After block N")}</option>
                                </select>
                              </div>

                              {(p.anchor === "before_block" || p.anchor === "after_block") ? (
                                <div>
                                  <label className="text-[10px] text-[#a1a1aa] block">Block N (1-{Math.max(1, totalBlocks)})</label>
                                  <input
                                    type="number"
                                    min="1"
                                    max={Math.max(1, totalBlocks)}
                                    value={p.anchorBlock}
                                    onChange={(e) =>
                                      handleUpdatePlacement(p.id, {
                                        anchorBlock: Math.max(1, parseInt(e.target.value, 10) || 1),
                                      })
                                    }
                                    className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                  />
                                </div>
                              ) : (
                                <div>
                                  <label className="text-[10px] text-[#a1a1aa] block">{t("Until block M (optional)")}</label>
                                  <input
                                    type="number"
                                    min="1"
                                    max={Math.max(1, totalBlocks)}
                                    value={p.untilBlock || ""}
                                    placeholder={t("None")}
                                    onChange={(e) => {
                                      const val = e.target.value.trim() ? parseInt(e.target.value, 10) : undefined;
                                      handleUpdatePlacement(p.id, { untilBlock: val });
                                    }}
                                    className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                  />
                                </div>
                              )}
                            </div>

                            {(p.anchor === "before_block" || p.anchor === "after_block") && (
                              <div>
                                <label className="text-[10px] text-[#a1a1aa] block">{t("Until block M (optional)")}</label>
                                <input
                                  type="number"
                                  min={p.anchorBlock}
                                  max={Math.max(1, totalBlocks)}
                                  value={p.untilBlock || ""}
                                  placeholder={t("None")}
                                  onChange={(e) => {
                                    const val = e.target.value.trim() ? parseInt(e.target.value, 10) : undefined;
                                    handleUpdatePlacement(p.id, { untilBlock: val });
                                  }}
                                  className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                />
                              </div>
                            )}

                            {/* Resolved span display */}
                            <div className="text-[11px] font-mono text-[#c4ff3d] bg-[#18181b] px-2 py-1 rounded">
                              {(() => {
                                const aType = p.anchor === "before_block" ? "before" : p.anchor === "after_block" ? "after" : p.anchor;
                                const span = resolvePlacementSpan({
                                  lineNum: 0,
                                  anchorType: aType as any,
                                  anchorBlock: p.anchorBlock,
                                  untilBlock: p.untilBlock,
                                  clipFrom: p.clipFrom,
                                  clipTo: p.clipTo,
                                  volumeDb: p.volumeDb,
                                  fadeIn: p.fadeIn,
                                  fadeOut: p.fadeOut,
                                  duckSpeech: p.duckSpeech,
                                  rawLine: "",
                                }, totalBlocks);
                                return span.description;
                              })()}
                            </div>

                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="text-[10px] text-[#a1a1aa] block">{t("Clip from (sec)")}</label>
                                <input
                                  type="number"
                                  min="0"
                                  max="30"
                                  value={p.clipFrom}
                                  onChange={(e) =>
                                    handleUpdatePlacement(p.id, { clipFrom: Math.max(0, parseFloat(e.target.value) || 0) })
                                  }
                                  className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                />
                              </div>
                              <div>
                                <label className="text-[10px] text-[#a1a1aa] block">{t("Clip to (sec)")}</label>
                                <input
                                  type="number"
                                  min="0"
                                  max="30"
                                  value={p.clipTo}
                                  onChange={(e) =>
                                    handleUpdatePlacement(p.id, { clipTo: Math.max(0, parseFloat(e.target.value) || 0) })
                                  }
                                  className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                />
                              </div>
                            </div>

                            <div className="space-y-1">
                              <div className="flex justify-between text-[10px] text-[#a1a1aa]">
                                <span>{t("Volume")}</span>
                                <span className="font-mono text-white">{p.volumeDb} dB</span>
                              </div>
                              <input
                                type="range"
                                min="-30"
                                max="0"
                                value={p.volumeDb}
                                onChange={(e) =>
                                  handleUpdatePlacement(p.id, { volumeDb: parseInt(e.target.value, 10) })
                                }
                                className="w-full accent-[#c4ff3d]"
                              />
                            </div>

                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <label className="text-[10px] text-[#a1a1aa] block">{t("Fade in (sec)")}</label>
                                <input
                                  type="number"
                                  min="0"
                                  step="0.5"
                                  value={p.fadeIn}
                                  onChange={(e) =>
                                    handleUpdatePlacement(p.id, { fadeIn: Math.max(0, parseFloat(e.target.value) || 0) })
                                  }
                                  className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                />
                              </div>
                              <div>
                                <label className="text-[10px] text-[#a1a1aa] block">{t("Fade out (sec)")}</label>
                                <input
                                  type="number"
                                  min="0"
                                  step="0.5"
                                  value={p.fadeOut}
                                  onChange={(e) =>
                                    handleUpdatePlacement(p.id, { fadeOut: Math.max(0, parseFloat(e.target.value) || 0) })
                                  }
                                  className="w-full bg-[#18181b] border border-white/[0.1] rounded px-1.5 py-1 text-[11px] text-white font-mono"
                                />
                              </div>
                            </div>

                            <label className="flex items-center gap-2 cursor-pointer pt-0.5">
                              <input
                                type="checkbox"
                                checked={p.duckSpeech}
                                onChange={(e) =>
                                  handleUpdatePlacement(p.id, { duckSpeech: e.target.checked })
                                }
                                className="w-3 h-3 text-[#c4ff3d] accent-[#c4ff3d]"
                              />
                              <span className="text-[11px] text-[#a1a1aa]">{t("Duck speech (-6 dB during speech)")}</span>
                            </label>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Card 5: Create podcast */}
            <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                onClick={() => setOpenCreatePodcast(!openCreatePodcast)}
              >
                <div className="flex items-center gap-2.5">
                  <Sparkles className="w-4 h-4 text-[#c4ff3d]" />
                  <span className="text-sm font-semibold text-white tracking-wide">{t("Create podcast")}</span>
                </div>
                <button
                  type="button"
                  aria-expanded={openCreatePodcast}
                  aria-controls="card-create-podcast-body"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenCreatePodcast(!openCreatePodcast);
                  }}
                  className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                  title={t(openCreatePodcast ? "Collapse" : "Expand")}
                >
                  <ChevronDown
                    className={`w-4 h-4 transform transition-transform duration-200 ${openCreatePodcast ? "rotate-180" : ""}`}
                  />
                </button>
              </div>

              {openCreatePodcast && (
                <div id="card-create-podcast-body" className="p-4 sm:p-5 pt-0 space-y-4 border-t border-white/[0.06]">
                  {/* Create podcast button with format choice next to it */}
                  <div className="flex items-center gap-2.5">
                    <button
                      type="button"
                      onClick={handleSynthesize}
                      disabled={
                        synthesizing ||
                        requestMode ||
                        writingScript ||
                        validationErrors.length > 0 ||
                        totalBlocks === 0
                      }
                      className="flex-1 py-3 px-4 text-xs sm:text-sm font-semibold text-[#0a0a0b] bg-[#c4ff3d] rounded-xl hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(196,255,61,0.25)]"
                    >
                      {synthesizing ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>
                            {progress?.message || t("Synthesizing ({done}/{total})...", {
                              done: progress?.chunksDone || 0,
                              total: progress?.chunksTotal || chunksCount,
                            })}
                          </span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4 fill-current" />
                          <span>{t("Create podcast")}</span>
                        </>
                      )}
                    </button>

                    <div className="flex items-center p-1 bg-[#18181b] border border-white/[0.1] rounded-xl shrink-0">
                      {(["mp3", "wav"] as const).map((fmt) => {
                        const isSelected = audioOutput.format === fmt;
                        return (
                          <button
                            key={fmt}
                            type="button"
                            onClick={() => handleAudioOutputChange({ format: fmt })}
                            className={`px-3 py-1.5 text-xs font-mono font-semibold rounded-lg uppercase transition-all ${
                              isSelected
                                ? "bg-[#c4ff3d] text-[#0a0a0b] shadow-sm"
                                : "text-[#a1a1aa] hover:text-white"
                            }`}
                          >
                            {fmt}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Live Synthesis Progress */}
                  {synthesizing && progress && (
                    <div className="space-y-2.5 pt-1">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-[#c4ff3d] flex items-center gap-1.5 truncate max-w-[200px]">
                          <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                          {progress.message}
                        </span>
                        <span className="text-[#a1a1aa] shrink-0">
                          {t("Chunk")} {progress.chunksDone} / {progress.chunksTotal}
                        </span>
                      </div>
                      <div className="w-full bg-[#18181b] rounded-full h-2 overflow-hidden border border-white/[0.05]">
                        <div
                          className="bg-[#c4ff3d] h-2 transition-all duration-300 shadow-[0_0_10px_rgba(196,255,61,0.5)]"
                          style={{
                            width: `${
                              progress.chunksTotal > 0
                                ? Math.round((progress.chunksDone / progress.chunksTotal) * 100)
                                : 0
                            }%`,
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Synthesis Error Display */}
                  {synthError && (
                    <div className="p-3 bg-red-950/40 border border-red-800/80 rounded-xl text-xs text-red-200 flex items-start justify-between gap-2 font-mono">
                      <div>
                        <span className="font-semibold text-red-300">{t("Synthesis Error:")} </span>
                        {synthError}
                      </div>
                      <button
                        type="button"
                        onClick={() => setSynthError(null)}
                        className="text-red-400 hover:text-white underline text-[11px] shrink-0"
                      >
                        {t("Dismiss")}
                      </button>
                    </div>
                  )}

                  {/* Audio Player / Result Display */}
                  {synthResult ? (
                    <div className="space-y-4 pt-1 border-t border-white/[0.06]">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Volume2 className="w-4 h-4 text-[#c4ff3d]" />
                          <span className="text-xs font-semibold text-white">{t("Audio Player")}</span>
                        </div>
                        <span className="text-xs px-2.5 py-0.5 bg-[#c4ff3d]/15 text-[#c4ff3d] border border-[#c4ff3d]/30 rounded-full font-mono font-medium">
                          {synthResult.durationSeconds.toFixed(1)}s
                        </span>
                      </div>

                      {/* Styled Waveform Visualization */}
                      <div className="p-3 bg-[#18181b] border border-white/[0.08] rounded-xl flex flex-col gap-3">
                        <div className="flex items-center justify-between gap-1 h-12 px-1 overflow-hidden">
                          {[
                            30, 60, 45, 80, 100, 70, 50, 85, 95, 60, 40, 75, 90, 55, 35, 70,
                            85, 100, 65, 45, 80, 95, 70, 50, 30, 60, 85, 75, 40, 65, 90, 55,
                          ].map((height, i) => (
                            <span
                              key={i}
                              className="flex-1 bg-[#c4ff3d]/70 rounded-full transition-all hover:bg-[#c4ff3d]"
                              style={{ height: `${height}%` }}
                            />
                          ))}
                        </div>

                        {/* Native HTML5 Audio Element */}
                        <audio
                          controls
                          src={synthResult.url}
                          className="w-full h-9 accent-[#c4ff3d] rounded-lg"
                        />
                      </div>

                      {/* Audio Metadata */}
                      <div className="text-xs font-mono space-y-1.5 text-[#a1a1aa]">
                        <div className="truncate text-white font-medium">
                          {synthResult.fileName}
                        </div>
                        <div>Model: {effectiveModel}</div>
                        <div className="flex flex-wrap gap-x-2 gap-y-1 text-white/90">
                          <span>{t("Format")}: <strong className="text-[#c4ff3d] font-semibold uppercase">{synthResult.format || audioOutput.format}</strong></span>
                          <span>·</span>
                          <span>{t("Rate:")} <strong className="text-white font-medium">{synthResult.sampleRate || audioOutput.sampleRate} Hz</strong></span>
                          <span>·</span>
                          <span>{t("Channels")}: <strong className="text-white font-medium capitalize">{synthResult.channels || audioOutput.channels}</strong></span>
                          {(synthResult.format || audioOutput.format) === "mp3" && (
                            <>
                              <span>·</span>
                              <span>{t("Bitrate:")} <strong className="text-[#c4ff3d] font-medium">{synthResult.mp3Bitrate || audioOutput.mp3Bitrate} kbps</strong></span>
                            </>
                          )}
                        </div>
                        <div>{t("Duration:")} <strong className="text-white font-medium">{synthResult.durationSeconds.toFixed(1)}s</strong></div>
                        <div className="text-[11px] text-[#52525b]">
                          {t("Completed")} {synthResult.chunkLogs.length} {t("chunks")}
                        </div>
                      </div>

                      {/* Download Button */}
                      <a
                        href={synthResult.url}
                        download={synthResult.fileName}
                        className="w-full py-2.5 px-4 text-xs font-semibold text-[#0a0a0b] bg-[#c4ff3d] rounded-xl hover:brightness-110 transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(196,255,61,0.25)]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        {t("Download")} {(synthResult.format || audioOutput.format).toUpperCase()}
                      </a>
                    </div>
                  ) : (
                    <div className="p-4 bg-[#18181b] border border-white/[0.06] rounded-xl space-y-3 text-center">
                      <div className="flex items-center justify-between gap-1 h-10 px-2 opacity-30">
                        {[
                          25, 40, 60, 35, 50, 70, 45, 60, 80, 55, 40, 65, 50, 35, 60, 45,
                          70, 55, 40, 60, 50, 35, 55, 40, 30, 45, 60, 50, 35, 40, 55, 30,
                        ].map((height, i) => (
                          <span
                            key={i}
                            className="flex-1 bg-white rounded-full"
                            style={{ height: `${height}%` }}
                          />
                        ))}
                      </div>
                      <p className="text-xs text-[#a1a1aa] leading-relaxed">
                        {t("Click Generate Podcast to synthesize audio and display the waveform player.")}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Card 6: Advanced (closed by default) */}
            <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                onClick={() => setOpenAdvanced(!openAdvanced)}
              >
                <div className="flex items-center gap-2.5">
                  <Sliders className="w-4 h-4 text-[#c4ff3d]" />
                  <span className="text-sm font-semibold text-white tracking-wide">{t("Advanced")}</span>
                </div>
                <button
                  type="button"
                  aria-expanded={openAdvanced}
                  aria-controls="card-advanced-body"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenAdvanced(!openAdvanced);
                  }}
                  className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                  title={t(openAdvanced ? "Collapse" : "Expand")}
                >
                  <ChevronDown
                    className={`w-4 h-4 transform transition-transform duration-200 ${openAdvanced ? "rotate-180" : ""}`}
                  />
                </button>
              </div>

              {openAdvanced && (
                <div id="card-advanced-body" className="p-4 sm:p-5 pt-0 space-y-4 border-t border-white/[0.06]">
                  <div className="space-y-3">
                    <p className="text-xs text-[#a1a1aa] leading-relaxed">
                      Configure lead-in and tail background music buffers applied before speech begins and after speech ends.
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <label className="text-xs font-semibold text-white block">{t("Music lead-in (sec)")}</label>
                        <input
                          type="number"
                          min="0"
                          step="0.5"
                          value={audioOutput.musicLeadIn}
                          onChange={(e) =>
                            handleAudioOutputChange({
                              musicLeadIn: Math.max(0, parseFloat(e.target.value) || 0),
                            })
                          }
                          className="w-full bg-[#18181b] border border-white/[0.1] rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-[#c4ff3d]"
                        />
                        <span className="text-[10px] text-[#71717a] block font-mono">{t("Before block 1")}</span>
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs font-semibold text-white block">{t("Music tail (sec)")}</label>
                        <input
                          type="number"
                          min="0"
                          step="0.5"
                          value={audioOutput.musicTail}
                          onChange={(e) =>
                            handleAudioOutputChange({
                              musicTail: Math.max(0, parseFloat(e.target.value) || 0),
                            })
                          }
                          className="w-full bg-[#18181b] border border-white/[0.1] rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-[#c4ff3d]"
                        />
                        <span className="text-[10px] text-[#71717a] block font-mono">{t("After last block")}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 4. Footer: Tools & Reference collapsible card (collapsed by default) */}
        <footer className="w-full mt-2">
          <div className="bg-[#121214] border border-white/[0.08] rounded-[20px] shadow-xl overflow-hidden">
            <div
              className="w-full flex items-center justify-between p-4 sm:p-5 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
              onClick={() => setOpenToolsRef(!openToolsRef)}
            >
              <div className="flex items-center gap-2.5">
                <FileCode className="w-4 h-4 text-[#c4ff3d]" />
                <span className="text-sm font-semibold text-white tracking-wide">
                  {t("Tools & Reference")}
                </span>
              </div>
              <button
                type="button"
                aria-expanded={openToolsRef}
                aria-controls="card-tools-ref-body"
                onClick={(e) => {
                  e.stopPropagation();
                  setOpenToolsRef(!openToolsRef);
                }}
                className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                title={t(openToolsRef ? "Collapse" : "Expand")}
              >
                <ChevronDown
                  className={`w-4 h-4 transform transition-transform duration-200 ${openToolsRef ? "rotate-180" : ""}`}
                />
              </button>
            </div>

            {openToolsRef && (
              <div id="card-tools-ref-body" className="p-4 sm:p-6 pt-0 space-y-4 border-t border-white/[0.06]">
                {/* Sub-card a: Usage & Quota (Today) */}
                <div className="bg-[#18181b] border border-white/[0.08] rounded-xl overflow-hidden">
                  <div
                    className="w-full flex items-center justify-between px-4 py-3 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                    onClick={() => setOpenUsage(!openUsage)}
                  >
                    <div className="flex items-center gap-2">
                      <Gauge className="w-3.5 h-3.5 text-[#c4ff3d]" />
                      <span className="text-xs font-semibold text-white">{t("Usage & Quota (Today)")}</span>
                    </div>
                    <button
                      type="button"
                      aria-expanded={openUsage}
                      aria-controls="card-usage-body"
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenUsage(!openUsage);
                      }}
                      className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                      title={t(openUsage ? "Collapse" : "Expand")}
                    >
                      <ChevronDown
                        className={`w-3.5 h-3.5 transform transition-transform duration-200 ${openUsage ? "rotate-180" : ""}`}
                      />
                    </button>
                  </div>

                  {openUsage && (
                    <div id="card-usage-body" className="p-4 pt-2 space-y-3 border-t border-white/[0.06]">
                      {(() => {
                        const reqPercent = Math.min(
                          100,
                          Math.round((todayUsage.requests / usageLimits.dailyRequests) * 100)
                        );
                        const wordPercent = Math.min(
                          100,
                          Math.round((todayUsage.words / usageLimits.dailyWords) * 100)
                        );

                        const formatNum = (num: number | string) =>
                          selectedLang === "fa" ? toPersianDigits(num) : String(num);

                        const getBarColor = (pct: number) => {
                          if (pct >= 90) return "bg-red-500";
                          if (pct >= 70) return "bg-amber-400";
                          return "bg-[#c4ff3d]";
                        };

                        const getStatusText = (pct: number) => {
                          if (pct >= 90)
                            return { label: "Limit almost reached", color: "text-red-400" };
                          if (pct >= 70)
                            return { label: "Near limit", color: "text-amber-400" };
                          return { label: "OK", color: "text-[#c4ff3d]" };
                        };

                        const remRequests = Math.max(0, usageLimits.dailyRequests - todayUsage.requests);
                        const remWords = Math.max(0, usageLimits.dailyWords - todayUsage.words);

                        return (
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-white">{t("Today's Summary")}</span>
                              <span className="text-[11px] text-[#a1a1aa] font-mono">
                                {getTodayKey().replace("usage_", "")}
                              </span>
                            </div>

                            {/* Requests Progress Bar */}
                            <div className="space-y-1">
                              <div className="flex items-center justify-between text-xs font-mono">
                                <span className="text-[#a1a1aa]">{t("Requests today")}</span>
                                <span className="text-white font-medium">
                                  {formatNum(todayUsage.requests)} / {formatNum(usageLimits.dailyRequests)} ({formatNum(reqPercent)}%)
                                </span>
                              </div>
                              <div className="w-full bg-[#121214] border border-white/[0.08] rounded-full h-[6px] overflow-hidden">
                                <div
                                  className={`h-[6px] rounded-full transition-all duration-300 ${getBarColor(reqPercent)}`}
                                  style={{ width: `${reqPercent}%` }}
                                />
                              </div>
                              <div className="flex items-center justify-between text-[11px] font-mono">
                                <span className={getStatusText(reqPercent).color}>
                                  {getStatusText(reqPercent).label}
                                </span>
                                <span className="text-[#71717a]">
                                  Remaining: {formatNum(remRequests)}
                                </span>
                              </div>
                            </div>

                            {/* Words Progress Bar */}
                            <div className="space-y-1">
                              <div className="flex items-center justify-between text-xs font-mono">
                                <span className="text-[#a1a1aa]">{t("Words today")}</span>
                                <span className="text-white font-medium">
                                  {formatNum(todayUsage.words)} / {formatNum(usageLimits.dailyWords)} ({formatNum(wordPercent)}%)
                                </span>
                              </div>
                              <div className="w-full bg-[#121214] border border-white/[0.08] rounded-full h-[6px] overflow-hidden">
                                <div
                                  className={`h-[6px] rounded-full transition-all duration-300 ${getBarColor(wordPercent)}`}
                                  style={{ width: `${wordPercent}%` }}
                                />
                              </div>
                              <div className="flex items-center justify-between text-[11px] font-mono">
                                <span className={getStatusText(wordPercent).color}>
                                  {getStatusText(wordPercent).label}
                                </span>
                                <span className="text-[#71717a]">
                                  Remaining: {formatNum(remWords)}
                                </span>
                              </div>
                            </div>

                            {/* Remaining amount summary */}
                            <div className="p-2.5 bg-[#121214] border border-white/[0.05] rounded-lg text-[11px] font-mono space-y-1 text-[#a1a1aa]">
                              <div className="text-white">
                                Remaining today: {formatNum(remRequests)} requests, {formatNum(remWords)} words.
                              </div>
                              <div className="text-[#71717a] flex items-center justify-between">
                                <span>Session: {formatNum(sessionRequests)} requests</span>
                                <span>Audio today: {formatNum(Math.round(todayUsage.audioSeconds))}s</span>
                              </div>
                            </div>

                            {/* Editable Daily Limits */}
                            <div className="p-2.5 bg-[#121214] border border-white/[0.06] rounded-lg space-y-2">
                              <div className="text-[11px] font-semibold text-white">{t("Daily Limits")}</div>
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <label className="text-[11px] text-[#a1a1aa] block mb-1">{t("Daily request limit")}</label>
                                  <input
                                    type="number"
                                    min="1"
                                    value={usageLimits.dailyRequests}
                                    onChange={(e) => {
                                      const rawVal = e.target.value.trim();
                                      const val = parseInt(rawVal, 10);
                                      const updated = {
                                        ...usageLimits,
                                        dailyRequests: isNaN(val) || val <= 0 ? 1500 : val,
                                      };
                                      setUsageLimits(updated);
                                      saveUsageLimits(updated);
                                    }}
                                    className="w-full bg-[#18181b] border border-white/[0.1] rounded px-2 py-1 text-xs text-white font-mono focus:outline-none focus:border-[#c4ff3d]"
                                  />
                                </div>
                                <div>
                                  <label className="text-[11px] text-[#a1a1aa] block mb-1">{t("Daily word limit")}</label>
                                  <input
                                    type="number"
                                    min="1"
                                    value={usageLimits.dailyWords}
                                    onChange={(e) => {
                                      const rawVal = e.target.value.trim();
                                      const val = parseInt(rawVal, 10);
                                      const updated = {
                                        ...usageLimits,
                                        dailyWords: isNaN(val) || val <= 0 ? 100000 : val,
                                      };
                                      setUsageLimits(updated);
                                      saveUsageLimits(updated);
                                    }}
                                    className="w-full bg-[#18181b] border border-white/[0.1] rounded px-2 py-1 text-xs text-white font-mono focus:outline-none focus:border-[#c4ff3d]"
                                  />
                                </div>
                              </div>
                            </div>

                            {/* Official Quota Link & Note */}
                            <div className="pt-2 border-t border-white/[0.06] space-y-2 text-[11px] text-[#a1a1aa] leading-relaxed">
                              <p>
                                These numbers are counted by this app only. The official quota may be lower.
                              </p>

                              <div className="flex items-center justify-between gap-2 pt-0.5">
                                <a
                                  href="https://ai.dev/rate-limit"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-[#0a0a0b] bg-[#c4ff3d] rounded-lg hover:brightness-110 transition-all font-sans"
                                >
                                  <span>{t("Check official quota")}</span>
                                  <ExternalLink className="w-3 h-3" />
                                </a>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>

                {/* Sub-card b: Python CLI */}
                <div className="bg-[#18181b] border border-white/[0.08] rounded-xl overflow-hidden">
                  <div
                    className="w-full flex items-center justify-between px-4 py-3 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                    onClick={() => setOpenCli(!openCli)}
                  >
                    <div className="flex items-center gap-2">
                      <FileCode className="w-3.5 h-3.5 text-[#c4ff3d]" />
                      <span className="text-xs font-semibold text-white">{t("Python CLI")}</span>
                    </div>
                    <button
                      type="button"
                      aria-expanded={openCli}
                      aria-controls="card-cli-body"
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenCli(!openCli);
                      }}
                      className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                      title={t(openCli ? "Collapse" : "Expand")}
                    >
                      <ChevronDown
                        className={`w-3.5 h-3.5 transform transition-transform duration-200 ${openCli ? "rotate-180" : ""}`}
                      />
                    </button>
                  </div>

                  {openCli && (
                    <div id="card-cli-body" className="p-4 pt-2 space-y-4 border-t border-white/[0.06]">
                      <div className="border border-white/[0.08] bg-[#121214] rounded-xl overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/[0.08] bg-[#18181b]">
                          <div className="flex items-center gap-2 text-xs font-mono text-[#f5f5f7]">
                            <FileCode className="w-4 h-4 text-[#c4ff3d]" />
                            <span>make_podcast.py</span>
                          </div>
                          <button
                            type="button"
                            onClick={handleCopyCli}
                            className="px-3 py-1 text-xs font-medium text-[#f5f5f7] bg-[#121214] hover:bg-white/[0.08] border border-white/[0.08] rounded-lg transition-colors flex items-center gap-1.5"
                          >
                            {copiedCli ? (
                              <>
                                <Check className="w-3.5 h-3.5 text-[#c4ff3d]" />
                                {t("Copied")}
                              </>
                            ) : (
                              <>
                                <Copy className="w-3.5 h-3.5 text-[#a1a1aa]" />
                                {t("Copy Script")}
                              </>
                            )}
                          </button>
                        </div>
                        <pre className="p-4 text-xs font-mono text-[#f5f5f7] overflow-x-auto max-h-[480px] leading-relaxed">
                          <code>{pythonScript}</code>
                        </pre>
                      </div>

                      <div className="p-4 bg-[#121214] border border-white/[0.08] rounded-xl space-y-3">
                        <h4 className="text-xs font-semibold text-white flex items-center gap-2">
                          <Radio className="w-3.5 h-3.5 text-[#c4ff3d]" />
                          {t("CLI Usage")}
                        </h4>
                        <p className="text-xs text-[#a1a1aa] leading-relaxed">
                          {t("Run the Python script with --lang for any supported language:")}
                        </p>
                        <pre className="p-3 bg-[#18181b] border border-white/[0.06] rounded-lg text-xs font-mono text-[#f5f5f7] overflow-x-auto leading-relaxed">
{`python make_podcast.py --lang en
python make_podcast.py --lang fa
python make_podcast.py --lang ar
python make_podcast.py --lang en --model ${DEFAULT_TTS_MODEL} --voice1 Rami --voice2 Leda`}
                        </pre>
                        <p className="text-[11px] text-[#52525b]">
                          {t("Outputs to")} <code className="text-[#c4ff3d]">outputs/podcast_&lt;lang&gt;_&lt;timestamp&gt;.wav</code>.
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Sub-card c: Technical Specs */}
                <div className="bg-[#18181b] border border-white/[0.08] rounded-xl overflow-hidden">
                  <div
                    className="w-full flex items-center justify-between px-4 py-3 cursor-pointer select-none hover:bg-white/[0.02] transition-colors"
                    onClick={() => setOpenSpecs(!openSpecs)}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="w-3.5 h-3.5 text-[#c4ff3d]" />
                      <span className="text-xs font-semibold text-white">{t("Technical Specs")}</span>
                    </div>
                    <button
                      type="button"
                      aria-expanded={openSpecs}
                      aria-controls="card-specs-body"
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenSpecs(!openSpecs);
                      }}
                      className="p-1 rounded-lg text-[#a1a1aa] hover:text-white hover:bg-white/[0.08] transition-colors focus:outline-none"
                      title={t(openSpecs ? "Collapse" : "Expand")}
                    >
                      <ChevronDown
                        className={`w-3.5 h-3.5 transform transition-transform duration-200 ${openSpecs ? "rotate-180" : ""}`}
                      />
                    </button>
                  </div>

                  {openSpecs && (
                    <div id="card-specs-body" className="p-4 pt-2 space-y-4 border-t border-white/[0.06]">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-[#121214] border border-white/[0.08] rounded-xl space-y-3">
                          <div className="flex items-center gap-2 text-xs text-[#c4ff3d] font-mono">
                            <Layers className="w-3.5 h-3.5" />
                            <span>{t("Audio Pipeline & Storage")}</span>
                          </div>
                          <h4 className="text-sm font-semibold text-white">
                            {t("Direct Browser Synthesis & RIFF WAV Builder")}
                          </h4>
                          <p className="text-xs text-[#a1a1aa] leading-relaxed">
                            {t("Audio is streamed chunk-by-chunk in 20-block batches via Gemini TTS API, formatted with a 44-byte RIFF WAVE header, and converted into a standard playable and downloadable WAV Blob.")}
                          </p>
                          <div className="border-t border-white/[0.08] pt-2 space-y-1.5 text-xs font-mono">
                            <div className="flex justify-between">
                              <span className="text-[#a1a1aa]">{t("Format")}</span>
                              <span className="text-[#f5f5f7]">{t("16-bit Linear PCM RIFF WAV")}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[#a1a1aa]">{t("Sample Rate")}</span>
                              <span className="text-[#f5f5f7]">{t("24,000 Hz Mono")}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[#a1a1aa]">{t("Engine")}</span>
                              <span className="text-[#c4ff3d]">{t("Client Gemini Flash TTS")}</span>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-[#121214] border border-white/[0.08] rounded-xl space-y-3">
                          <div className="text-xs text-[#c4ff3d] font-mono">
                            {t("Languages & Batching")}
                          </div>
                          <h4 className="text-sm font-semibold text-white">
                            {t("Multilingual Dual-Speaker Support")}
                          </h4>
                          <p className="text-xs text-[#a1a1aa] leading-relaxed">
                            {t("Supports English, Persian, and Arabic with fixed voices: Speaker 1 = Rami, Speaker 2 = Leda.")}
                          </p>
                          <div className="border-t border-white/[0.08] pt-2 space-y-1.5 text-xs font-mono">
                            <div className="flex justify-between">
                              <span className="text-[#a1a1aa]">{t("Chunk Size")}</span>
                              <span className="text-[#f5f5f7]">{t("20 blocks")}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[#a1a1aa]">{t("Retry Attempts")}</span>
                              <span className="text-[#f5f5f7]">{t("3 retries (5s wait)")}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[#a1a1aa]">{t("Coverage Gates")}</span>
                              <span className="text-[#c4ff3d]">{t("0.6 chunk / 0.85 total")}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </footer>
      </main>
    </div>
  );
}
