# Gemini Podcast Audio Generator

A production-grade two-speaker podcast generation application and Python CLI pipeline powered by the Gemini TTS model (`gemini-3.8-flash-tts`) and `@google/genai` / `google-genai` SDKs.

## Features

- **Full-Script Textarea & Live Validation**: Edit and paste full podcast scripts with live line validation, debounced syntax checks, block counts, and error highlighting.
- **Multilingual Support**: Supports English (`en`), Persian (`fa`), and Arabic (`ar`) via language-specific script files (`script.en.txt`, `script.fa.txt`, `script.ar.txt`). Auto-direction (`dir="auto"` and RTL for Persian/Arabic) with unsaved change warnings.
- **Fixed Voice Roles**:
  - **Speaker 1 (Arash)**: Prebuilt voice `Rami`
  - **Speaker 2 (Sara)**: Prebuilt voice `Leda`
  - Conversational speech synthesis mode (`CONVERSATIONAL`).
- **Intermediate WAV & MP3 Output**: Saves audio files with timestamps in `outputs/podcast_<lang>_<timestamp>.(wav|mp3)` and converts them to 64 kbps 24 kHz mono MP3 via FFmpeg.
- **Robust 20-Block Batching & Progress**: Batches dialogue in chunks of 20 with up to 3 retries (5s wait) and real-time status polling (`/api/synthesize/status`).
- **Clean Duration & State**: State transitions use `idle`, `in_progress`, `done`, or `error`. Clean duration display shows `—` before synthesis and exact server duration after synthesis.

---

## Run Locally

### 1. Environment Configuration

Create a `.env.local` file in the project root and add your Gemini API key:

```bash
# .env.local (preferred; falls back to .env)
GEMINI_API_KEY="your-gemini-api-key-here"
```

The server automatically loads `.env.local` first and falls back to `.env`:
```typescript
dotenv.config({ path: ".env.local" });
dotenv.config();
```

If `GEMINI_API_KEY` is missing at startup, a warning is printed to the console, and `/api/synthesize` returns a clean JSON error response rather than crashing.

### 2. Node.js Full-Stack App

1. Install dependencies:
   ```bash
   npm install
   ```

2. Build and start the development server:
   ```bash
   npm run build
   npm run dev
   ```

3. Open `http://localhost:3000` in your browser.

### 3. Standalone Python CLI (`make_podcast.py`)

Run the podcast generator directly from the command line with `--lang` (default `en`):

```bash
pip install google-genai
export GEMINI_API_KEY="your-gemini-api-key-here"

# English (default)
python make_podcast.py --lang en

# Persian
python make_podcast.py --lang fa

# Arabic
python make_podcast.py --lang ar
```

The script parses `script.<lang>.txt`, chunks dialogue into 20-block requests, and saves the final WAV audio into `outputs/podcast_<lang>_<timestamp>.wav`.

---

## Endpoints

| Method | Path | Query / Body | Response |
|---|---|---|---|
| GET | `/api/script` | `?lang=en\|fa\|ar` | `{ lang, blocks, raw }` |
| POST | `/api/script` | `{ lang, content }` | `{ ok, count, lang }` or error |
| GET | `/api/synthesize/status` | — | `{ state, chunksDone, chunksTotal, message }` |
| POST | `/api/synthesize` | `{ lang, blocks }` | `{ fileName, url, durationSeconds, chunkLogs, warning? }` |

---

## FFmpeg Fallback Behavior

If `ffmpeg` is missing on the host system (`ENOENT` error) or conversion fails:
- The intermediate high-fidelity WAV file is preserved in `outputs/`.
- No audio data is silently lost.
- HTTP 200 is returned with `url` pointing to the playable WAV file and an explanatory `warning` message.
- The UI shows an amber notice banner above the audio player while keeping the audio player fully playable.
