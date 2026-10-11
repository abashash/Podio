// Audio processing, WAV generation, MP3 encoding, and Web Audio mixer

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import lamejs from "lamejs";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import lameAllCode from "lamejs/lame.all.js?raw";
import { countWords } from "./usage";

export interface AudioOutputConfig {
  format: "wav" | "mp3";
  sampleRate: 24000 | 16000 | 44100;
  channels: "mono" | "stereo";
  mp3Bitrate: 64 | 96 | 128 | 192;
  gainDb: number; // -12 dB to +12 dB
  trimSilence: boolean;
  musicLeadIn: number; // Lead-in silence before first speech block (sec, default 0)
  musicTail: number;   // Tail silence after last speech block (sec, default 0)
}

export const DEFAULT_AUDIO_OUTPUT: AudioOutputConfig = {
  format: "mp3",
  sampleRate: 24000,
  channels: "mono",
  mp3Bitrate: 128,
  gainDb: 0,
  trimSilence: false,
  musicLeadIn: 0,
  musicTail: 0,
};

export function getAudioOutputConfig(): AudioOutputConfig {
  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return DEFAULT_AUDIO_OUTPUT;
    }
    const raw = localStorage.getItem("audio_output");
    if (!raw) return DEFAULT_AUDIO_OUTPUT;
    const parsed = JSON.parse(raw);
    return {
      format: parsed.format === "mp3" ? "mp3" : "wav",
      sampleRate: [16000, 24000, 44100].includes(parsed.sampleRate) ? parsed.sampleRate : 24000,
      channels: parsed.channels === "stereo" ? "stereo" : "mono",
      mp3Bitrate: [64, 96, 128, 192].includes(parsed.mp3Bitrate) ? parsed.mp3Bitrate : 128,
      gainDb: typeof parsed.gainDb === "number" && !isNaN(parsed.gainDb) ? Math.min(12, Math.max(-12, parsed.gainDb)) : 0,
      trimSilence: Boolean(parsed.trimSilence),
      musicLeadIn: typeof parsed.musicLeadIn === "number" && !isNaN(parsed.musicLeadIn) && parsed.musicLeadIn >= 0 ? parsed.musicLeadIn : 0,
      musicTail: typeof parsed.musicTail === "number" && !isNaN(parsed.musicTail) && parsed.musicTail >= 0 ? parsed.musicTail : 0,
    };
  } catch {
    return DEFAULT_AUDIO_OUTPUT;
  }
}

export function saveAudioOutputConfig(config: AudioOutputConfig): void {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      localStorage.setItem("audio_output", JSON.stringify(config));
    }
  } catch (err) {
    console.warn("Failed to save audio_output to localStorage:", err);
  }
}

export function parseAudioMimeType(mimeType: string): {
  bitsPerSample: number;
  rate: number;
} {
  let bitsPerSample = 16;
  let rate = 24000;

  if (!mimeType) {
    return { bitsPerSample, rate };
  }

  const parts = mimeType.split(";");
  for (const rawParam of parts) {
    const param = rawParam.trim();
    const lower = param.toLowerCase();
    if (lower.startsWith("rate=")) {
      const parsed = Number.parseInt(param.split("=")[1] ?? "", 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        rate = parsed;
      }
    } else if (lower.startsWith("audio/l")) {
      const bitStr = param.slice("audio/l".length);
      const parsed = Number.parseInt(bitStr, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        bitsPerSample = parsed;
      }
    }
  }

  return { bitsPerSample, rate };
}

// Convert base64 string to Uint8Array
export function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// If Gemini returns a WAV file containing a RIFF header, extract the raw data bytes
export function extractPcmPayload(rawBuffer: Uint8Array): Uint8Array {
  if (rawBuffer.length >= 44) {
    const headerStr = String.fromCharCode(...rawBuffer.subarray(0, 4));
    const waveStr = String.fromCharCode(...rawBuffer.subarray(8, 12));
    if (headerStr === "RIFF" && waveStr === "WAVE") {
      const view = new DataView(
        rawBuffer.buffer,
        rawBuffer.byteOffset,
        rawBuffer.byteLength
      );
      let offset = 12;
      while (offset + 8 <= rawBuffer.length) {
        const subchunkId = String.fromCharCode(...rawBuffer.subarray(offset, offset + 4));
        const subchunkSize = view.getUint32(offset + 4, true);
        if (subchunkId === "data") {
          const start = offset + 8;
          const end = Math.min(start + subchunkSize, rawBuffer.length);
          return rawBuffer.subarray(start, end);
        }
        offset += 8 + subchunkSize;
      }
      return rawBuffer.subarray(44);
    }
  }
  return rawBuffer;
}

// Concatenate multiple Uint8Arrays into one
export function concatUint8Arrays(arrays: Uint8Array[]): Uint8Array {
  let totalLength = 0;
  for (const arr of arrays) {
    totalLength += arr.length;
  }
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

// Convert 16-bit little-endian PCM bytes to Float32Array [-1.0, 1.0]
export function pcm16ToFloat32(pcmBytes: Uint8Array): Float32Array {
  const sampleCount = Math.floor(pcmBytes.length / 2);
  const floats = new Float32Array(sampleCount);
  const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
  for (let i = 0; i < sampleCount; i++) {
    const int16 = view.getInt16(i * 2, true);
    floats[i] = int16 < 0 ? int16 / 32768 : int16 / 32767;
  }
  return floats;
}

// Convert Float32Array [-1.0, 1.0] back to 16-bit little-endian Int16Array
export function float32ToInt16(floats: Float32Array): Int16Array {
  const len = floats.length;
  const int16 = new Int16Array(len);
  for (let i = 0; i < len; i++) {
    let s = Math.max(-1, Math.min(1, floats[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

// Trim silence at beginning and end (threshold approx -45 dB = 0.005)
export function trimSilenceFloat32(samples: Float32Array, threshold = 0.005): Float32Array {
  let start = 0;
  while (start < samples.length && Math.abs(samples[start]) < threshold) {
    start++;
  }
  let end = samples.length - 1;
  while (end > start && Math.abs(samples[end]) < threshold) {
    end--;
  }
  if (start >= end) return samples;
  return samples.subarray(start, end + 1);
}

// Apply volume gain in dB
export function applyGainFloat32(samples: Float32Array, gainDb: number): Float32Array {
  if (gainDb === 0) return samples;
  const factor = Math.pow(10, gainDb / 20);
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    out[i] = Math.max(-1, Math.min(1, samples[i] * factor));
  }
  return out;
}

// Resample Float32 mono audio from srcRate to dstRate using OfflineAudioContext
export async function resampleFloat32Mono(
  samples: Float32Array,
  srcRate: number,
  dstRate: number
): Promise<Float32Array> {
  if (srcRate === dstRate) return samples;
  const targetLength = Math.max(1, Math.round((samples.length * dstRate) / srcRate));
  const offlineCtx = new (window.OfflineAudioContext || (window as any).webkitOfflineAudioContext)(
    1,
    targetLength,
    dstRate
  );

  const buffer = offlineCtx.createBuffer(1, samples.length, srcRate);
  buffer.copyToChannel(samples as any, 0);

  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(offlineCtx.destination);
  source.start(0);

  const renderedBuffer = await offlineCtx.startRendering();
  return renderedBuffer.getChannelData(0);
}

// Create a WAV Blob from raw PCM bytes and MIME type info (legacy default)
export function createWavBlob(rawPcmData: Uint8Array, mimeType: string): Blob {
  const { bitsPerSample, rate } = parseAudioMimeType(mimeType);
  const channels = 1;
  const byteRate = (rate * channels * bitsPerSample) / 8;
  const blockAlign = (channels * bitsPerSample) / 8;
  const dataLength = rawPcmData.length;

  const header = new ArrayBuffer(44);
  const view = new DataView(header);

  // Write "RIFF"
  view.setUint8(0, 0x52);
  view.setUint8(1, 0x49);
  view.setUint8(2, 0x46);
  view.setUint8(3, 0x46);
  view.setUint32(4, 36 + dataLength, true);

  // Write "WAVE"
  view.setUint8(8, 0x57);
  view.setUint8(9, 0x41);
  view.setUint8(10, 0x56);
  view.setUint8(11, 0x45);

  // Write "fmt "
  view.setUint8(12, 0x66);
  view.setUint8(13, 0x6d);
  view.setUint8(14, 0x74);
  view.setUint8(15, 0x20);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM = 1
  view.setUint16(22, channels, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // Write "data"
  view.setUint8(36, 0x64);
  view.setUint8(37, 0x61);
  view.setUint8(38, 0x74);
  view.setUint8(39, 0x61);
  view.setUint32(40, dataLength, true);

  return new Blob([header, rawPcmData as any], { type: "audio/wav" });
}

// Build a WAV Blob with custom channels and sample rate from Int16Array samples
export function buildCustomWavBlob(
  leftSamples: Int16Array,
  rightSamples: Int16Array | null,
  sampleRate: number
): Blob {
  const isStereo = rightSamples !== null;
  const numChannels = isStereo ? 2 : 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const numFrames = leftSamples.length;
  const dataByteLength = numFrames * blockAlign;

  const header = new ArrayBuffer(44);
  const view = new DataView(header);

  // RIFF header
  view.setUint8(0, 0x52); // R
  view.setUint8(1, 0x49); // I
  view.setUint8(2, 0x46); // F
  view.setUint8(3, 0x46); // F
  view.setUint32(4, 36 + dataByteLength, true);

  // WAVE
  view.setUint8(8, 0x57);  // W
  view.setUint8(9, 0x41);  // A
  view.setUint8(10, 0x56); // V
  view.setUint8(11, 0x45); // E

  // fmt
  view.setUint8(12, 0x66);
  view.setUint8(13, 0x6d);
  view.setUint8(14, 0x74);
  view.setUint8(15, 0x20);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data
  view.setUint8(36, 0x64);
  view.setUint8(37, 0x61);
  view.setUint8(38, 0x74);
  view.setUint8(39, 0x61);
  view.setUint32(40, dataByteLength, true);

  // Interleave PCM data into byte buffer
  const pcmBuffer = new ArrayBuffer(dataByteLength);
  const pcmView = new DataView(pcmBuffer);
  let offset = 0;

  if (isStereo && rightSamples) {
    for (let i = 0; i < numFrames; i++) {
      pcmView.setInt16(offset, leftSamples[i], true);
      pcmView.setInt16(offset + 2, rightSamples[i], true);
      offset += 4;
    }
  } else {
    for (let i = 0; i < numFrames; i++) {
      pcmView.setInt16(offset, leftSamples[i], true);
      offset += 2;
    }
  }

  return new Blob([header, pcmBuffer], { type: "audio/wav" });
}

let cachedLameInstance: any = null;
export function getMp3EncoderClass(): any {
  if (cachedLameInstance?.Mp3Encoder) return cachedLameInstance.Mp3Encoder;
  try {
    if (typeof window !== "undefined" && (window as any).lamejs?.Mp3Encoder) {
      cachedLameInstance = (window as any).lamejs;
      return cachedLameInstance.Mp3Encoder;
    }
    if (typeof lameAllCode === "string" && lameAllCode.length > 0) {
      const fn = new Function(lameAllCode + "; return lamejs;");
      cachedLameInstance = fn();
      return cachedLameInstance.Mp3Encoder;
    }
    // Node.js fallback for test runner / scripts
    if (typeof process !== "undefined" && typeof require !== "undefined") {
      try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const fs = require("fs");
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const path = require("path");
        const p = path.resolve("./node_modules/lamejs/lame.all.js");
        if (fs.existsSync(p)) {
          const code = fs.readFileSync(p, "utf8");
          const fn = new Function(code + "; return lamejs;");
          cachedLameInstance = fn();
          return cachedLameInstance.Mp3Encoder;
        }
      } catch {}
    }
  } catch (err) {
    console.warn("Could not instantiate lamejs from bundled source:", err);
  }
  try {
    if (lamejs && typeof lamejs.Mp3Encoder === "function") {
      return lamejs.Mp3Encoder;
    }
  } catch {}
  return null;
}

// Encode PCM samples to MP3 using lamejs
export function encodeMp3(
  leftSamples: Int16Array,
  rightSamples: Int16Array | null,
  sampleRate: number,
  bitrateKbps: number
): Blob {
  const Mp3Encoder = getMp3EncoderClass();
  if (!Mp3Encoder) {
    throw new Error("MP3 encoder (lamejs) is not available. Please select WAV format.");
  }
  const isStereo = rightSamples !== null;
  const channels = isStereo ? 2 : 1;
  const mp3encoder = new Mp3Encoder(channels, sampleRate, bitrateKbps);
  const mp3Data: Uint8Array[] = [];

  const sampleBlockSize = 1152;
  const numSamples = leftSamples.length;

  for (let i = 0; i < numSamples; i += sampleBlockSize) {
    const end = Math.min(i + sampleBlockSize, numSamples);
    const leftChunk = leftSamples.subarray(i, end);
    let mp3buf: Int8Array | Uint8Array;

    if (isStereo && rightSamples) {
      const rightChunk = rightSamples.subarray(i, end);
      mp3buf = mp3encoder.encodeBuffer(leftChunk, rightChunk);
    } else {
      mp3buf = mp3encoder.encodeBuffer(leftChunk);
    }

    if (mp3buf.length > 0) {
      mp3Data.push(new Uint8Array(mp3buf.buffer, mp3buf.byteOffset, mp3buf.length));
    }
  }

  const mp3buf = mp3encoder.flush();
  if (mp3buf.length > 0) {
    mp3Data.push(new Uint8Array(mp3buf.buffer, mp3buf.byteOffset, mp3buf.length));
  }

  return new Blob(mp3Data as BlobPart[], { type: "audio/mp3" });
}

// Mix speech Float32 with background music file using Web Audio API OfflineAudioContext
export async function mixSpeechWithMusic(
  speechFloat32: Float32Array,
  speechRate: number,
  musicFile: File | Blob,
  musicVolumeDb = -18,
  loopMusic = true
): Promise<Float32Array> {
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  const musicArrayBuffer = await musicFile.arrayBuffer();
  const musicAudioBuffer = await audioCtx.decodeAudioData(musicArrayBuffer);
  await audioCtx.close();

  const speechDuration = speechFloat32.length / speechRate;
  const targetSampleRate = speechRate;
  const totalFrames = speechFloat32.length;

  const offlineCtx = new (window.OfflineAudioContext || (window as any).webkitOfflineAudioContext)(
    1,
    totalFrames,
    targetSampleRate
  );

  // Speech source
  const speechBuffer = offlineCtx.createBuffer(1, speechFloat32.length, speechRate);
  speechBuffer.copyToChannel(speechFloat32 as any, 0);
  const speechSource = offlineCtx.createBufferSource();
  speechSource.buffer = speechBuffer;
  speechSource.connect(offlineCtx.destination);
  speechSource.start(0);

  // Music source
  const musicSource = offlineCtx.createBufferSource();
  musicSource.buffer = musicAudioBuffer;
  musicSource.loop = loopMusic;

  const musicGain = offlineCtx.createGain();
  const musicGainFactor = Math.pow(10, musicVolumeDb / 20);
  musicGain.gain.setValueAtTime(musicGainFactor, 0);

  // Subtle fade out for music at the end of podcast
  if (speechDuration > 1.5) {
    musicGain.gain.setValueAtTime(musicGainFactor, speechDuration - 1.0);
    musicGain.gain.linearRampToValueAtTime(0, speechDuration);
  }

  musicSource.connect(musicGain);
  musicGain.connect(offlineCtx.destination);
  musicSource.start(0);

  const rendered = await offlineCtx.startRendering();
  return rendered.getChannelData(0);
}

// Process full speech pipeline with optional resampling, gain, trim, stereo, music, and encoding (WAV or MP3)
export async function processPodcastAudioOutput(
  rawPcmData: Uint8Array,
  detectedMimeType: string,
  config: AudioOutputConfig,
  musicFile: File | null = null,
  musicVolumeDb = -18,
  loopMusic = true
): Promise<{ blob: Blob; mimeType: string; durationSeconds: number; extension: "wav" | "mp3" }> {
  const { rate: baseRate } = parseAudioMimeType(detectedMimeType);

  // 1. Convert to Float32
  let floats = pcm16ToFloat32(rawPcmData);

  // 2. Trim silence if requested
  if (config.trimSilence) {
    floats = trimSilenceFloat32(floats);
  }

  // 3. Apply volume gain
  if (config.gainDb !== 0) {
    floats = applyGainFloat32(floats, config.gainDb);
  }

  // 4. Mix with background music if provided
  if (musicFile) {
    try {
      floats = await mixSpeechWithMusic(floats, baseRate, musicFile, musicVolumeDb, loopMusic);
    } catch (err) {
      console.warn("Failed to mix background music, proceeding with speech only:", err);
    }
  }

  // 5. Resample if config.sampleRate differs from baseRate
  const targetRate = config.sampleRate || 24000;
  if (targetRate !== baseRate) {
    floats = await resampleFloat32Mono(floats, baseRate, targetRate);
  }

  // 6. Convert to 16-bit PCM
  const leftInt16 = float32ToInt16(floats);
  const rightInt16 = config.channels === "stereo" ? new Int16Array(leftInt16) : null;
  const durationSeconds = Number((leftInt16.length / targetRate).toFixed(2));

  // 7. Encode according to config.format
  if (config.format === "mp3") {
    try {
      const mp3Blob = encodeMp3(leftInt16, rightInt16, targetRate, config.mp3Bitrate || 128);
      return { blob: mp3Blob, mimeType: "audio/mp3", durationSeconds, extension: "mp3" };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`MP3 encoding failed: ${msg}. You can switch to WAV format.`);
    }
  }

  const wavBlob = buildCustomWavBlob(leftInt16, rightInt16, targetRate);
  return { blob: wavBlob, mimeType: "audio/wav", durationSeconds, extension: "wav" };
}

// Background Music and Lyria 3 Placement Types
export interface MusicClip {
  id: string;
  clipNumber: number;
  base64: string;
  audioUrl: string;
  durationSeconds: number;
}

export type MusicAnchor = "start" | "end" | "before_block" | "after_block";

export interface MusicPlacement {
  id: string;
  clipId: string;
  anchor: MusicAnchor;
  anchorBlock: number;
  untilBlock?: number;
  clipFrom: number;
  clipTo: number;
  volumeDb: number;
  fadeIn: number;
  fadeOut: number;
  duckSpeech: boolean;
}

export interface BlockTiming {
  blockIndex: number;
  startTime: number;
  endTime: number;
  duration: number;
  wordCount: number;
}

// Compute each block's start and end time in the speech timeline from real per-chunk audio durations
export function computeBlockTimings(
  blocks: { text: string; lineNum?: number }[],
  chunkLogs: { chunkIndex: number; firstBlock: number; lastBlock: number; pcmBytes: number }[],
  sampleRate = 24000
): BlockTiming[] {
  const timings: BlockTiming[] = [];
  let chunkStartTime = 0;

  for (const log of chunkLogs) {
    const chunkDuration = log.pcmBytes / (sampleRate * 2); // 16-bit mono PCM = 2 bytes per sample
    const chunkBlocks = blocks.slice(log.firstBlock - 1, log.lastBlock);
    const wordCounts = chunkBlocks.map((b) => countWords(b.text || ""));
    const totalWords = wordCounts.reduce((acc, w) => acc + w, 0);

    let offsetInChunk = 0;
    chunkBlocks.forEach((_, idx) => {
      const blockNum = log.firstBlock + idx;
      const words = wordCounts[idx];
      const blockDuration =
        totalWords > 0
          ? chunkDuration * (words / totalWords)
          : chunkDuration / Math.max(1, chunkBlocks.length);

      const startTime = chunkStartTime + offsetInChunk;
      const endTime = startTime + blockDuration;
      offsetInChunk += blockDuration;

      timings.push({
        blockIndex: blockNum,
        startTime: Number(startTime.toFixed(3)),
        endTime: Number(endTime.toFixed(3)),
        duration: Number(blockDuration.toFixed(3)),
        wordCount: words,
      });
    });

    chunkStartTime += chunkDuration;
  }

  return timings;
}

// Calculate anchor start time and cutoff duration on speech timeline
export function computePlacementTiming(
  placement: MusicPlacement,
  blockTimings: BlockTiming[],
  totalSpeechDuration: number
): { startTime: number; maxDuration: number } {
  const clipLength = Math.max(0, placement.clipTo - placement.clipFrom);
  let startTime = 0;

  switch (placement.anchor) {
    case "start":
      startTime = 0;
      break;
    case "end":
      startTime = Math.max(0, Number((totalSpeechDuration - clipLength).toFixed(3)));
      break;
    case "before_block": {
      const timing = blockTimings.find((b) => b.blockIndex === placement.anchorBlock);
      startTime = timing ? timing.startTime : 0;
      break;
    }
    case "after_block": {
      const timing = blockTimings.find((b) => b.blockIndex === placement.anchorBlock);
      startTime = timing ? timing.endTime : 0;
      break;
    }
    default:
      startTime = 0;
  }

  let maxDuration = clipLength;
  if (placement.untilBlock) {
    const untilTiming = blockTimings.find((b) => b.blockIndex === placement.untilBlock);
    if (untilTiming) {
      const untilTime = untilTiming.startTime;
      if (untilTime > startTime) {
        maxDuration = Math.min(clipLength, untilTime - startTime);
      }
    }
  }

  return { startTime, maxDuration };
}

// Calculate anchor start time on speech timeline (legacy alias)
export function computePlacementStartTime(
  placement: MusicPlacement,
  blockTimings: BlockTiming[],
  totalSpeechDuration: number
): number {
  return computePlacementTiming(placement, blockTimings, totalSpeechDuration).startTime;
}

// Validation rules for placement
export function validatePlacement(
  p: MusicPlacement,
  totalBlocks: number,
  speechDuration: number
): { errors: string[]; isLongerThanClip: boolean } {
  const errors: string[] = [];
  if (p.clipFrom < 0 || p.clipFrom > 30 || p.clipTo < 0 || p.clipTo > 30) {
    errors.push("Clip range must be between 0 and 30 seconds");
  }
  if (p.clipTo <= p.clipFrom) {
    errors.push("To must be greater than From");
  }
  if (
    (p.anchor === "before_block" || p.anchor === "after_block") &&
    (p.anchorBlock < 1 || p.anchorBlock > totalBlocks)
  ) {
    errors.push("Anchor block number is out of range");
  }
  const placementLen = p.clipTo - p.clipFrom;
  if (speechDuration > 0 && placementLen > speechDuration) {
    errors.push("Placement is longer than the podcast; it will be cut");
  }
  const isLongerThanClip = placementLen > 30;
  return { errors, isLongerThanClip };
}

// Decodes base64 MP3 to 24000 Hz mono Float32Array using Web Audio API / OfflineAudioContext
export async function decodeBase64ClipToMono24k(
  base64: string,
  targetRate = 24000
): Promise<Float32Array> {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  if (typeof window !== "undefined" && (window.AudioContext || (window as any).webkitAudioContext)) {
    const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioCtx = new AudioCtxClass();
    try {
      const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer.slice(0));
      const targetFrames = Math.max(1, Math.round(audioBuffer.duration * targetRate));
      const OfflineCtxClass =
        window.OfflineAudioContext || (window as any).webkitOfflineAudioContext;
      const offlineCtx = new OfflineCtxClass(1, targetFrames, targetRate);
      const src = offlineCtx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(offlineCtx.destination);
      src.start(0);
      const rendered = await offlineCtx.startRendering();
      return rendered.getChannelData(0);
    } finally {
      if (audioCtx.state !== "closed") {
        audioCtx.close().catch(() => {});
      }
    }
  }

  // Fallback buffer
  const fallbackFrames = targetRate * 30;
  return new Float32Array(fallbackFrames);
}

// Mix speech with music placements using lead-in/tail, per-block modes (normal, duck -12dB, silent),
// until cuts, smooth 0.15s linear transitions, fades, and clamping.
export async function mixPodcastWithMusicPlacements(params: {
  speechPcm: Uint8Array;
  clips: MusicClip[];
  placements: MusicPlacement[];
  blockTimings: BlockTiming[];
  musicModes?: { startBlock: number; endBlock: number; mode: "normal" | "duck" | "silent" }[];
  leadInSeconds?: number;
  tailSeconds?: number;
  outputFormat?: "wav" | "mp3";
  channels?: "mono" | "stereo";
  mp3Bitrate?: number;
}): Promise<{
  blob: Blob;
  mimeType: string;
  durationSeconds: number;
  extension: "wav" | "mp3";
}> {
  const sampleRate = 24000;
  const leadIn = Math.max(0, params.leadInSeconds || 0);
  const tail = Math.max(0, params.tailSeconds || 0);

  const speechFloats = pcm16ToFloat32(params.speechPcm);
  const speechFrames = speechFloats.length;
  const totalSpeechDuration = speechFrames / sampleRate;

  const leadInFrames = Math.round(leadIn * sampleRate);
  const tailFrames = Math.round(tail * sampleRate);
  const totalFrames = leadInFrames + speechFrames + tailFrames;

  // Initialize mix buffer with silence, and insert speech at offset leadInFrames
  const mixBuffer = new Float32Array(totalFrames);
  mixBuffer.set(speechFloats, leadInFrames);

  // If there are no placements, return speech-only (with lead-in/tail if specified)
  if (!params.placements || params.placements.length === 0) {
    const leftInt16 = float32ToInt16(mixBuffer);
    const isStereo = params.channels === "stereo";
    const rightInt16 = isStereo ? new Int16Array(leftInt16) : null;
    const durationSeconds = Number((totalFrames / sampleRate).toFixed(2));
    const format = params.outputFormat || "wav";
    if (format === "mp3") {
      const mp3Blob = encodeMp3(leftInt16, rightInt16, sampleRate, params.mp3Bitrate || 128);
      return { blob: mp3Blob, mimeType: "audio/mp3", durationSeconds, extension: "mp3" };
    }
    const wavBlob = buildCustomWavBlob(leftInt16, rightInt16, sampleRate);
    return { blob: wavBlob, mimeType: "audio/wav", durationSeconds, extension: "wav" };
  }

  // Pre-calculate per-block mode mapping: blockIndex (1-based) -> mode
  // Overlap precedence: silent beats duck, duck beats normal. Default: duck if placement.duckSpeech is on, else normal.
  const blockModeMap = new Map<number, "normal" | "duck" | "silent">();
  if (params.musicModes && params.musicModes.length > 0) {
    for (const m of params.musicModes) {
      for (let b = m.startBlock; b <= m.endBlock; b++) {
        const cur = blockModeMap.get(b);
        if (!cur) {
          blockModeMap.set(b, m.mode);
        } else if (cur === "silent" || m.mode === "silent") {
          blockModeMap.set(b, "silent");
        } else if (cur === "duck" || m.mode === "duck") {
          blockModeMap.set(b, "duck");
        } else {
          blockModeMap.set(b, m.mode);
        }
      }
    }
  }

  // Helper to determine target mode at timeline second on the speech timeline
  const getModeAtSpeechSec = (sec: number, defaultDuck: boolean): "normal" | "duck" | "silent" => {
    // Lead-in and tail zones (sec < 0 or sec > totalSpeechDuration) are considered gaps -> normal
    if (sec < 0 || sec > totalSpeechDuration) return "normal";

    let matchingBlock: BlockTiming | undefined;
    for (const bt of params.blockTimings) {
      if (sec >= bt.startTime && sec <= bt.endTime) {
        matchingBlock = bt;
        break;
      }
    }

    if (!matchingBlock) {
      // Gap between blocks -> normal
      return "normal";
    }

    // Check if block has explicit mode
    const explicit = blockModeMap.get(matchingBlock.blockIndex);
    if (explicit) return explicit;

    return defaultDuck ? "duck" : "normal";
  };

  // Cache decoded 24kHz mono Float32 audio for each clip
  const clipBuffers = new Map<string, Float32Array>();
  for (const clip of params.clips) {
    if (!clipBuffers.has(clip.id)) {
      const floats = await decodeBase64ClipToMono24k(clip.base64, sampleRate);
      clipBuffers.set(clip.id, floats);
    }
  }

  // Mix each placement
  for (const placement of params.placements) {
    const clipFloats =
      clipBuffers.get(placement.clipId) ||
      (params.clips[0] ? clipBuffers.get(params.clips[0].id) : null);
    if (!clipFloats) continue;

    const clipDuration = Math.max(0, placement.clipTo - placement.clipFrom);
    if (clipDuration <= 0) continue;

    const timing = computePlacementTiming(placement, params.blockTimings, totalSpeechDuration);
    const speechStartTime = timing.startTime;
    const effectivePlacementDuration = Math.min(clipDuration, timing.maxDuration);
    if (effectivePlacementDuration <= 0) continue;

    // Timeline start in output timeline
    // If start anchor and leadIn > 0, start placement at beginning of leadIn (output time = 0)
    // If end anchor and tail > 0, placement plays during tail at the end
    let outputStartTimeSec: number;
    let speechRefTimeSec: number;

    if (placement.anchor === "start") {
      outputStartTimeSec = 0; // Plays during lead-in
      speechRefTimeSec = -leadIn;
    } else if (placement.anchor === "end") {
      const totalOutDuration = totalFrames / sampleRate;
      outputStartTimeSec = Math.max(0, totalOutDuration - effectivePlacementDuration);
      speechRefTimeSec = outputStartTimeSec - leadIn;
    } else {
      outputStartTimeSec = leadIn + speechStartTime;
      speechRefTimeSec = speechStartTime;
    }

    const startFrame = Math.round(outputStartTimeSec * sampleRate);
    const clipStartFrame = Math.round(placement.clipFrom * sampleRate);
    const placementFrames = Math.round(effectivePlacementDuration * sampleRate);

    const normalVolDb = placement.volumeDb;
    const duckVolDb = placement.volumeDb - 12; // -12 dB per spec for duck mode

    const normalGain = Math.pow(10, normalVolDb / 20);
    const duckGain = Math.pow(10, duckVolDb / 20);
    const silentGain = 0;

    const transitionFrames = Math.max(1, Math.round(0.15 * sampleRate)); // 0.15s linear transitions

    // Pre-calculate target gain per sample, smoothing transitions
    let currentGain = normalGain;

    for (let k = 0; k < placementFrames; k++) {
      const dstIdx = startFrame + k;
      if (dstIdx >= totalFrames) break; // Cut at end of output

      const srcIdx = clipStartFrame + k;
      if (srcIdx >= clipFloats.length) break;

      const tSec = k / sampleRate;
      const currentSpeechSec = speechRefTimeSec + tSec;

      // Determine target mode at current speech second
      const mode = getModeAtSpeechSec(currentSpeechSec, placement.duckSpeech);
      let targetModeGain: number;
      if (mode === "silent") {
        targetModeGain = silentGain;
      } else if (mode === "duck") {
        targetModeGain = duckGain;
      } else {
        targetModeGain = normalGain;
      }

      // Smooth step towards targetModeGain (0.15s transition)
      if (currentGain < targetModeGain) {
        currentGain = Math.min(targetModeGain, currentGain + (targetModeGain - currentGain) / transitionFrames);
      } else if (currentGain > targetModeGain) {
        currentGain = Math.max(targetModeGain, currentGain - (currentGain - targetModeGain) / transitionFrames);
      }

      // Placement fade in and fade out
      let fadeMultiplier = 1.0;
      if (placement.fadeIn > 0 && tSec < placement.fadeIn) {
        fadeMultiplier = tSec / placement.fadeIn;
      } else if (
        placement.fadeOut > 0 &&
        tSec > effectivePlacementDuration - placement.fadeOut
      ) {
        fadeMultiplier = Math.max(
          0,
          (effectivePlacementDuration - tSec) / placement.fadeOut
        );
      }

      const finalGain = currentGain * fadeMultiplier;
      mixBuffer[dstIdx] += clipFloats[srcIdx] * finalGain;
    }
  }

  // Clamp sum to [-1.0, 1.0]
  for (let i = 0; i < totalFrames; i++) {
    mixBuffer[i] = Math.max(-1.0, Math.min(1.0, mixBuffer[i]));
  }

  // Convert to 16-bit PCM
  const leftInt16 = float32ToInt16(mixBuffer);
  const isStereo = params.channels === "stereo";
  const rightInt16 = isStereo ? new Int16Array(leftInt16) : null;
  const durationSeconds = Number((totalFrames / sampleRate).toFixed(2));

  // Encode
  const format = params.outputFormat || "wav";
  if (format === "mp3") {
    try {
      const mp3Blob = encodeMp3(leftInt16, rightInt16, sampleRate, params.mp3Bitrate || 128);
      return { blob: mp3Blob, mimeType: "audio/mp3", durationSeconds, extension: "mp3" };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`MP3 encoding failed: ${msg}. You can switch to WAV format.`);
    }
  }

  const wavBlob = buildCustomWavBlob(leftInt16, rightInt16, sampleRate);
  return { blob: wavBlob, mimeType: "audio/wav", durationSeconds, extension: "wav" };
}
