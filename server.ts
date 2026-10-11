import express, { type Request } from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import {
  DEFAULT_TTS_MODEL,
  DEFAULT_VOICES,
  MUSIC_MODEL,
  TEXT_MODEL,
} from "./src/config.ts";
import { parseScriptDirectives } from "./src/directives.ts";
import { createTranslator, normalizeLocale } from "./src/i18n.ts";

const execFileAsync = promisify(execFile);

// Load .env.local first, falling back to .env
dotenv.config({ path: ".env.local" });
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

if (!process.env.GEMINI_API_KEY) {
  console.error("WARNING: GEMINI_API_KEY is missing. Please define it in .env.local or .env.");
}

const MODEL_NAME = DEFAULT_TTS_MODEL;
const CHUNK_SIZE = 20;
const MAX_RETRIES = 3;
const RETRY_WAIT_MS = 5000;

function getTranslator(req: Request): (key: string) => string {
  const acceptLanguage = req.get("accept-language")?.split(",")[0]?.split("-")[0];
  return createTranslator(normalizeLocale(req.body?.lang ?? req.query.lang ?? acceptLanguage));
}

function translateMessage(
  t: (key: string) => string,
  key: string,
  values: Record<string, string | number> = {}
): string {
  return Object.entries(values).reduce(
    (message, [name, value]) => message.replaceAll(`{${name}}`, String(value)),
    t(key)
  );
}

export type SupportedLang = "en" | "fa" | "ar";

export interface PodcastBlock {
  speaker: "Rami" | "Leda" | string;
  style: string;
  text: string;
  lineNum?: number;
}

const SPEAKER_MAP: Record<string, string> = {
  Rami: "Speaker 1",
  Leda: "Speaker 2",
};

// Synthesis status: only "idle" | "in_progress" | "done" | "error"
export interface SynthesisStatus {
  state: "idle" | "in_progress" | "done" | "error";
  chunksDone: number;
  chunksTotal: number;
  message: string;
}

let synthesisStatus: SynthesisStatus = {
  state: "idle",
  chunksDone: 0,
  chunksTotal: 0,
  message: "Idle",
};

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

export function extractPcmPayload(rawBuffer: Buffer): Buffer {
  if (
    rawBuffer.length >= 44 &&
    rawBuffer.subarray(0, 4).toString("ascii") === "RIFF" &&
    rawBuffer.subarray(8, 12).toString("ascii") === "WAVE"
  ) {
    let offset = 12;
    while (offset + 8 <= rawBuffer.length) {
      const subchunkId = rawBuffer.subarray(offset, offset + 4).toString("ascii");
      const subchunkSize = rawBuffer.readUInt32LE(offset + 4);
      if (subchunkId === "data") {
        const start = offset + 8;
        const end = Math.min(start + subchunkSize, rawBuffer.length);
        return rawBuffer.subarray(start, end);
      }
      offset += 8 + subchunkSize;
    }
    return rawBuffer.subarray(44);
  }
  return rawBuffer;
}

export function convertToWav(rawPcmData: Buffer, mimeType: string): Buffer {
  const { bitsPerSample, rate } = parseAudioMimeType(mimeType);
  const channels = 1;
  const byteRate = (rate * channels * bitsPerSample) / 8;
  const blockAlign = (channels * bitsPerSample) / 8;
  const dataLength = rawPcmData.length;
  const buffer = Buffer.alloc(44 + dataLength);

  buffer.write("RIFF", 0, "ascii");
  buffer.writeUInt32LE(36 + dataLength, 4);
  buffer.write("WAVE", 8, "ascii");

  buffer.write("fmt ", 12, "ascii");
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM format
  buffer.writeUInt16LE(channels, 22);
  buffer.writeUInt32LE(rate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);

  buffer.write("data", 36, "ascii");
  buffer.writeUInt32LE(dataLength, 40);

  rawPcmData.copy(buffer, 44);
  return buffer;
}

let genAIInstance: GoogleGenAI | null = null;
function getGenAIClient(): GoogleGenAI {
  if (!genAIInstance) {
    genAIInstance = new GoogleGenAI();
  }
  return genAIInstance;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function getScriptFileName(lang: string): string {
  if (lang === "fa") return "script.fa.txt";
  if (lang === "ar") return "script.ar.txt";
  return "script.en.txt";
}

export function parseScriptText(content: string): {
  blocks: PodcastBlock[];
  errors: Array<{ lineNum: number; problem: string; preview: string }>;
} {
  const lines = content.split(/\r?\n/);
  const blocks: PodcastBlock[] = [];
  const errors: Array<{ lineNum: number; problem: string; preview: string }> = [];

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

    // Ignore placeholder comments starting with #
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

export async function readScriptForLang(lang: SupportedLang): Promise<{
  blocks: PodcastBlock[];
  raw: string;
}> {
  const fileName = getScriptFileName(lang);
  const scriptPath = path.join(__dirname, fileName);
  let raw = "";
  try {
    raw = await fs.readFile(scriptPath, "utf-8");
  } catch (err: any) {
    if (err.code === "ENOENT") {
      raw = "=== SCRIPT START ===\n=== SCRIPT END ===\n";
      await fs.writeFile(scriptPath, raw, "utf-8");
    } else {
      throw err;
    }
  }

  const { blocks, errors } = parseScriptText(raw);
  if (errors.length > 0) {
    const firstErr = errors[0];
    throw new Error(
      `Line ${firstErr.lineNum}: ${firstErr.problem}: "${firstErr.preview}"`
    );
  }
  return { blocks, raw };
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "10mb" }));

  const outputsDir = path.join(__dirname, "outputs");
  await fs.mkdir(outputsDir, { recursive: true });
  app.use("/outputs", express.static(outputsDir));

  // GET /api/script?lang=en|fa|ar
  app.get("/api/script", async (req, res) => {
    const t = getTranslator(req);
    try {
      const langParam = (req.query.lang as string) || "en";
      const validLangs: SupportedLang[] = ["en", "fa", "ar"];
      const lang: SupportedLang = validLangs.includes(langParam as SupportedLang)
        ? (langParam as SupportedLang)
        : "en";

      const { blocks, raw } = await readScriptForLang(lang);
      res.json({ lang, blocks, raw });
    } catch (error) {
      res.status(400).json({
        error: t("Failed to load script"),
      });
    }
  });

  // POST /api/script with body { lang, content }
  app.post("/api/script", async (req, res) => {
    const t = getTranslator(req);
    try {
      const { lang, content } = req.body as { lang?: string; content?: string };
      const validLangs: SupportedLang[] = ["en", "fa", "ar"];
      const targetLang: SupportedLang = validLangs.includes(lang as SupportedLang)
        ? (lang as SupportedLang)
        : "en";

      if (typeof content !== "string") {
        res.status(400).json({ error: t("content string is required.") });
        return;
      }

      const { blocks, errors } = parseScriptText(content);
      if (errors.length > 0) {
        const firstErr = errors[0];
        res.status(400).json({
          error: translateMessage(t, "Script validation failed at line {line}: {problem}", {
            line: firstErr.lineNum,
            problem: firstErr.problem,
          }),
          errors,
        });
        return;
      }

      const fileName = getScriptFileName(targetLang);
      const scriptPath = path.join(__dirname, fileName);
      await fs.writeFile(scriptPath, content, "utf-8");

      res.json({ ok: true, count: blocks.length, lang: targetLang });
    } catch (error) {
      res.status(500).json({
        error: t("Failed to write script file."),
      });
    }
  });

  // GET /api/workspace for backwards compatibility
  app.get("/api/workspace", async (req, res) => {
    const t = getTranslator(req);
    try {
      const [scriptContent, enScript] = await Promise.all([
        fs.readFile(path.join(__dirname, "make_podcast.py"), "utf-8"),
        readScriptForLang("en").catch(async () => {
          const raw = await fs.readFile(path.join(__dirname, "blocks.json"), "utf-8");
          return { blocks: JSON.parse(raw), raw };
        }),
      ]);
      res.json({ scriptContent, blocks: enScript.blocks });
    } catch (error) {
      res.status(500).json({
        error: t("Failed to load workspace files."),
      });
    }
  });

  // POST /api/blocks writes to blocks.edited.json
  app.post("/api/blocks", async (req, res) => {
    const t = getTranslator(req);
    try {
      const { blocks } = req.body as { blocks: PodcastBlock[] };
      if (!Array.isArray(blocks) || blocks.length === 0) {
        res.status(400).json({ error: t("blocks must be a non-empty array.") });
        return;
      }
      await fs.writeFile(
        path.join(__dirname, "blocks.edited.json"),
        JSON.stringify(blocks, null, 2) + "\n",
        "utf-8"
      );
      res.json({ ok: true, count: blocks.length, file: "blocks.edited.json" });
    } catch (error) {
      res.status(500).json({
        error: t("Failed to save blocks.edited.json."),
      });
    }
  });

  // GET /api/synthesize/status
  app.get("/api/synthesize/status", (req, res) => {
    const t = getTranslator(req);
    res.json({ ...synthesisStatus, message: t(synthesisStatus.message) });
  });

  // POST /api/script/generate - Draft and validate a podcast script from a short request.
  app.post("/api/script/generate", async (req, res) => {
    const t = getTranslator(req);
    const { request, lang } = req.body as { request?: string; lang?: string };
    if (!request?.trim()) {
      res.status(400).json({ error: t("A short request is required.") });
      return;
    }
    if (!process.env.GEMINI_API_KEY) {
      res.status(400).json({
        error: t("GEMINI_API_KEY is missing. Please define it in .env.local or .env."),
      });
      return;
    }

    const locale = normalizeLocale(lang);
    const languageName = { en: "English", fa: "Persian", ar: "Arabic" }[locale];
    const model = process.env.TEXT_MODEL?.trim() || TEXT_MODEL;
    const basePrompt = [
      `Write an editable two-speaker podcast script in ${languageName}. Follow any language explicitly named in the request.`,
      "Use only dialogue lines in the exact format: Rami | Style | spoken text or Leda | Style | spoken text.",
      "Do not include markdown, code fences, headings, or script markers.",
      "Estimate the dialogue length from the requested duration at about 140 spoken words per minute.",
      "Include these three valid music directives before the dialogue:",
      "#MUSIC_PROMPT: Instrumental background music that suits the podcast subject and language; no vocals.",
      "#MUSIC: anchor=start clip=0-8 volume=-18 fade=1,1 duck=on",
      "#MUSIC: anchor=end clip=22-30 volume=-18 fade=1,1 duck=off",
      "Alternate naturally between Rami and Leda, use concise style labels, and make the conversation coherent.",
      `User request: ${request.trim()}`,
    ].join("\n");

    let lastError: unknown;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await getGenAIClient().models.generateContent({
          model,
          contents: [
            {
              role: "user",
              parts: [{
                text: attempt === 0
                  ? basePrompt
                  : `${basePrompt}\n\nThe previous result was invalid: ${lastError instanceof Error ? lastError.message : String(lastError)}. Return a corrected complete script only.`,
              }],
            },
          ],
          config: {
            temperature: 0.7,
            responseMimeType: "text/plain",
          },
        });

        const script = response.text?.trim() || "";
        if (!script) {
          throw new Error("The text model returned an empty script.");
        }
        const parsed = parseScriptText(script);
        if (parsed.errors.length > 0 || parsed.blocks.length === 0) {
          const firstError = parsed.errors[0];
          throw new Error(firstError
            ? `Line ${firstError.lineNum}: ${firstError.problem}`
            : "The script contains no dialogue lines.");
        }
        const directives = parseScriptDirectives(script, parsed.blocks.length);
        if (directives.errors.length > 0 || !directives.musicPrompt) {
          throw new Error(directives.errors[0]?.problem || "The script has no music prompt.");
        }
        const hasStartMusic = directives.musicDirectives.some((d) => d.anchorType === "start");
        const hasEndMusic = directives.musicDirectives.some((d) => d.anchorType === "end");
        if (!hasStartMusic || !hasEndMusic) {
          throw new Error("The script must place music at both the start and end.");
        }

        res.json({ script, model });
        return;
      } catch (error) {
        lastError = error;
      }
    }

    console.error("Failed to generate a valid podcast script:", lastError);
    res.status(502).json({
      error: t("Could not draft a valid script. Check the request and try again."),
    });
  });

  // POST /api/music/generate - Generates 30s background music clip using Lyria 3 via Interactions API
  app.post("/api/music/generate", async (req, res) => {
    const t = getTranslator(req);
    const { prompt } = req.body as { prompt?: string };
    if (!prompt || !prompt.trim()) {
      res.status(400).json({ error: t("Music prompt is required.") });
      return;
    }

    const apiKey = process.env.GEMINI_API_KEY || "";
    if (!apiKey) {
      res.status(500).json({ error: t("GEMINI_API_KEY is not configured on the server.") });
      return;
    }

    try {
      const apiResp = await fetch("https://generativelanguage.googleapis.com/v1beta/interactions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
          "Api-Revision": "2026-05-20",
        },
        body: JSON.stringify({
          model: MUSIC_MODEL,
          input: prompt.trim(),
        }),
      });

      const data = await apiResp.json();
      if (!apiResp.ok) {
        res.status(apiResp.status).json(data);
        return;
      }

      let audioBase64: string | null = null;
      let mimeType = "audio/mp3";

      if (data.output_audio?.data) {
        audioBase64 = data.output_audio.data;
        if (data.output_audio.mime_type) mimeType = data.output_audio.mime_type;
      } else if (Array.isArray(data.steps)) {
        for (const step of data.steps) {
          if (step.type === "model_output" && Array.isArray(step.content)) {
            const audioPart = step.content.find((c: any) => c.type === "audio");
            if (audioPart?.data) {
              audioBase64 = audioPart.data;
              if (audioPart.mime_type) mimeType = audioPart.mime_type;
              break;
            }
          }
        }
      }

      if (!audioBase64) {
        res.status(502).json({ error: t("No audio data returned by Lyria music model.") });
        return;
      }

      res.json({
        audioBase64,
        mimeType,
        durationSeconds: 30,
        model: MUSIC_MODEL,
      });
    } catch (err: any) {
      console.error("Music generation failed:", err);
      res.status(500).json({ error: t("Music generation failed.") });
    }
  });

  // POST /api/synthesize with body { lang, blocks }
  app.post("/api/synthesize", async (req, res) => {
    const t = getTranslator(req);
    if (!process.env.GEMINI_API_KEY) {
      res.status(400).json({
        error: t("GEMINI_API_KEY is missing. Please define it in .env.local or .env."),
      });
      return;
    }

    try {
      const { lang: rawLang, blocks, model: reqModel, voice1: reqVoice1, voice2: reqVoice2 } = req.body as {
        lang?: string;
        blocks: PodcastBlock[];
        model?: string;
        voice1?: string;
        voice2?: string;
      };

      const selectedModel = reqModel || DEFAULT_TTS_MODEL;
      const modelDefaults = DEFAULT_VOICES[selectedModel] || { voice1: "Rami", voice2: "Leda" };
      const voice1 = reqVoice1 || modelDefaults.voice1;
      const voice2 = reqVoice2 || modelDefaults.voice2;

      const validLangs: SupportedLang[] = ["en", "fa", "ar"];
      const lang: SupportedLang = validLangs.includes(rawLang as SupportedLang)
        ? (rawLang as SupportedLang)
        : "en";

      if (!Array.isArray(blocks) || blocks.length === 0) {
        res.status(400).json({ error: t("Provide at least one dialogue block to synthesize.") });
        return;
      }

      // Re-validate blocks on server
      for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        const lineDisplay = b.lineNum ?? i + 1;
        if (!b.speaker || (b.speaker !== "Rami" && b.speaker !== "Leda")) {
          res.status(400).json({
            error: translateMessage(t, "Line {line}: Speaker must be exactly 'Rami' or 'Leda'", {
              line: lineDisplay,
            }),
          });
          return;
        }
        if (typeof b.text !== "string" || !b.text.trim()) {
          res.status(400).json({
            error: translateMessage(t, "Line {line}: Dialogue text is empty", {
              line: lineDisplay,
            }),
          });
          return;
        }
      }

      const totalBlocks = blocks.length;
      const totalChunks = Math.ceil(totalBlocks / CHUNK_SIZE);

      synthesisStatus = {
        state: "in_progress",
        chunksDone: 0,
        chunksTotal: totalChunks,
        message: translateMessage(t, "Starting synthesis ({done} of {total} chunks)", {
          done: 0,
          total: totalChunks,
        }),
      };

      const ai = getGenAIClient();
      const pcmChunks: Buffer[] = [];
      const chunkLogs: Array<{
        chunkIndex: number;
        firstBlock: number;
        lastBlock: number;
        pcmBytes: number;
        attempts: number;
      }> = [];
      let detectedMimeType = "audio/L16;rate=24000";

      for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        const startIdx = chunkIdx * CHUNK_SIZE;
        const endIdx = Math.min(startIdx + CHUNK_SIZE, totalBlocks);
        const chunkBlocks = blocks.slice(startIdx, endIdx);
        const firstBlockNum = startIdx + 1;
        const lastBlockNum = endIdx;

        // Never send style label to API; keep for UI and server logs only
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

        let chunkPcm: Buffer | null = null;
        let lastError: unknown = null;
        let usedAttempts = 0;

        for (let attempt = 1; attempt <= MAX_RETRIES + 1; attempt++) {
          usedAttempts = attempt;
          try {
            const response = await ai.models.generateContent({
              model: selectedModel,
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

            const extractedBuffers: Buffer[] = [];
            const candidateParts = response.candidates?.[0]?.content?.parts ?? [];
            for (const part of candidateParts) {
              if (part.inlineData?.data) {
                if (part.inlineData.mimeType) {
                  detectedMimeType = part.inlineData.mimeType;
                }
                const rawBuf = Buffer.from(part.inlineData.data, "base64");
                extractedBuffers.push(extractPcmPayload(rawBuf));
              }
            }

            if (extractedBuffers.length === 0) {
              throw new Error("Gemini TTS response did not contain audio bytes.");
            }

            chunkPcm = Buffer.concat(extractedBuffers);
            break;
          } catch (err) {
            lastError = err;
            if (attempt <= MAX_RETRIES) {
              console.warn(
                `Chunk ${chunkIdx + 1}/${totalChunks} (blocks ${firstBlockNum}-${lastBlockNum}) failed attempt ${attempt}/${MAX_RETRIES}. Retrying in ${RETRY_WAIT_MS}ms...`
              );
              await sleep(RETRY_WAIT_MS);
            } else {
              const errMsg = err instanceof Error ? err.message : String(err);
              synthesisStatus = {
                state: "error",
                chunksDone: chunkIdx,
                chunksTotal: totalChunks,
                message: translateMessage(t, "Chunk {chunk} failed after {retries} retries.", {
                  chunk: `${chunkIdx + 1}/${totalChunks}`,
                  retries: MAX_RETRIES,
                }),
              };
              res.status(502).json({
                error: `${translateMessage(t, "Chunk {chunk} failed after {retries} retries.", {
                  chunk: `${chunkIdx + 1}/${totalChunks}`,
                  retries: MAX_RETRIES,
                })} ${errMsg}`,
              });
              return;
            }
          }
        }

        if (chunkPcm) {
          pcmChunks.push(chunkPcm);
          chunkLogs.push({
            chunkIndex: chunkIdx + 1,
            firstBlock: firstBlockNum,
            lastBlock: lastBlockNum,
            pcmBytes: chunkPcm.length,
            attempts: usedAttempts,
          });

          synthesisStatus = {
            state: "in_progress",
            chunksDone: chunkIdx + 1,
            chunksTotal: totalChunks,
            message: translateMessage(t, "Completed chunk {done} of {total}", {
              done: chunkIdx + 1,
              total: totalChunks,
            }),
          };
        } else if (lastError) {
          throw lastError;
        }
      }

      const combinedPcm = Buffer.concat(pcmChunks);
      const wavBuffer = convertToWav(combinedPcm, detectedMimeType);
      const { bitsPerSample, rate } = parseAudioMimeType(detectedMimeType);
      const durationSeconds = Number(
        (combinedPcm.length / (rate * (bitsPerSample / 8))).toFixed(2)
      );

      await fs.mkdir(outputsDir, { recursive: true });
      const timestamp = Date.now();
      const wavFileName = `podcast_${lang}_${timestamp}.wav`;
      const mp3FileName = `podcast_${lang}_${timestamp}.mp3`;
      const wavPath = path.join(outputsDir, wavFileName);
      const mp3Path = path.join(outputsDir, mp3FileName);

      await fs.writeFile(wavPath, wavBuffer);

      let finalFileName = wavFileName;
      let finalUrl = `/outputs/${wavFileName}`;

      try {
        await execFileAsync("ffmpeg", [
          "-y",
          "-i",
          wavPath,
          "-ac",
          "1",
          "-ar",
          "24000",
          "-b:a",
          "64k",
          mp3Path,
        ]);
        finalFileName = mp3FileName;
        finalUrl = `/outputs/${mp3FileName}`;
      } catch (ffmpegErr: any) {
        console.error("FFmpeg conversion error:", ffmpegErr?.message || ffmpegErr);
        if (ffmpegErr && ffmpegErr.code === "ENOENT") {
          const warningMessage = t("ffmpeg is not installed; returning WAV.");
          synthesisStatus = {
            state: "done",
            chunksDone: totalChunks,
            chunksTotal: totalChunks,
            message: warningMessage,
          };
          res.status(200).json({
            fileName: wavFileName,
            url: `/outputs/${wavFileName}`,
            durationSeconds,
            chunkLogs,
            warning: warningMessage,
          });
          return;
        }

        const warningMessage = translateMessage(t, "MP3 conversion failed ({error}); returning WAV.", {
          error: ffmpegErr.message,
        });
        synthesisStatus = {
            state: "done",
            chunksDone: totalChunks,
            chunksTotal: totalChunks,
            message: warningMessage,
        };
        res.status(200).json({
          fileName: wavFileName,
          url: `/outputs/${wavFileName}`,
          durationSeconds,
          chunkLogs,
          warning: warningMessage,
        });
        return;
      }

      synthesisStatus = {
        state: "done",
        chunksDone: totalChunks,
        chunksTotal: totalChunks,
        message: t("Synthesis and MP3 conversion done"),
      };

      res.status(200).json({
        fileName: finalFileName,
        url: finalUrl,
        durationSeconds,
        chunkLogs,
      });
    } catch (error) {
      const errMsg = error instanceof Error
        ? error.message
        : t("Unexpected error during podcast synthesis.");
      synthesisStatus = {
        state: "error",
        chunksDone: 0,
        chunksTotal: 0,
        message: errMsg,
      };
      res.status(500).json({ error: errMsg });
    }
  });

  // Catch-all for unknown /api routes: return JSON 404, never fallback to HTML index.html
  app.all("/api/*", (req, res) => {
    res.status(404).json({ error: getTranslator(req)("Not found") });
  });

  const distPath = path.join(__dirname, "dist");
  let hasDist = false;
  try {
    await fs.access(path.join(distPath, "index.html"));
    hasDist = true;
  } catch {
    hasDist = false;
  }

  if (hasDist) {
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Fatal startup error:", err);
  process.exit(1);
});
