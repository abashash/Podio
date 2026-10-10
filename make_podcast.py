# Install dependencies:
#   pip install google-genai
#
# Run instructions:
#   export GEMINI_API_KEY="your-api-key-here"
#   python make_podcast.py --lang en
#   python make_podcast.py --lang fa
#   python make_podcast.py --lang ar

from __future__ import annotations

import argparse
import base64
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore


DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
CHUNK_SIZE = 20
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5

VOICES_BY_MODEL: dict[str, list[str]] = {
    DEFAULT_TTS_MODEL: ["Rami", "Leda", "Puck", "Charon", "Kore", "Fenrir", "Aoede"],
    "gemini-2.5-pro-preview-tts": ["Rami", "Leda", "Puck", "Charon", "Kore", "Fenrir", "Aoede"],
    "gemini-2.5-flash-preview-tts": ["Rami", "Leda", "Puck", "Charon", "Kore", "Fenrir", "Aoede"],
}

DEFAULT_VOICES: dict[str, dict[str, str]] = {
    DEFAULT_TTS_MODEL: {"voice1": "Rami", "voice2": "Leda"},
    "gemini-2.5-pro-preview-tts": {"voice1": "Rami", "voice2": "Leda"},
    "gemini-2.5-flash-preview-tts": {"voice1": "Rami", "voice2": "Leda"},
}

START_MARKER = "=== SCRIPT START ==="
END_MARKER = "=== SCRIPT END ==="

EXPECTED_BLOCKS_BY_LANG = {"en": 141}
MIN_BLOCKS_BY_LANG = {"en": 141, "fa": 1, "ar": 1}

# Fixed speaker mapping:
# "Rami" is Speaker 1, Arash, prebuilt voice "Rami"
# "Leda" is Speaker 2, Sara, prebuilt voice "Leda"
SPEAKER_MAP = {
    "Rami": "Speaker 1",
    "Leda": "Speaker 2",
}


def parse_script_directives(lines: list[str], total_blocks: int) -> dict[str, Any]:
    """Parses # directives between START_MARKER and END_MARKER."""
    music_prompt: str | None = None
    music_directives: list[dict[str, Any]] = []
    music_modes: list[dict[str, Any]] = []
    errors: list[str] = []

    has_start = any(line.strip() == START_MARKER for line in lines)
    has_end = any(line.strip() == END_MARKER for line in lines)
    use_markers = has_start and has_end
    inside = False

    for line_num, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if use_markers:
            if stripped == START_MARKER:
                inside = True
                continue
            if stripped == END_MARKER:
                inside = False
                break
            if not inside:
                continue

        if not stripped or not stripped.startswith("#"):
            continue

        if stripped.startswith("#MUSIC_PROMPT:"):
            prompt_text = stripped[len("#MUSIC_PROMPT:"):].strip()
            if not prompt_text:
                errors.append(f"Line {line_num}: MUSIC_PROMPT directive has empty prompt text")
            elif music_prompt is not None:
                errors.append(f"Line {line_num}: Only one #MUSIC_PROMPT directive is allowed")
            else:
                music_prompt = prompt_text
            continue

        if stripped.startswith("#MUSIC:"):
            content = stripped[len("#MUSIC:"):].strip()
            if re.search(r"\b(at|sec)=", content, re.IGNORECASE):
                errors.append(f"Line {line_num}: Time-based placement is not supported; use block anchors")
                continue

            tokens = content.split()
            params: dict[str, str] = {}
            parse_failed = False
            for tok in tokens:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    params[k.lower()] = v
                else:
                    errors.append(f"Line {line_num}: Invalid parameter format '{tok}'")
                    parse_failed = True
                    break
            if parse_failed:
                continue

            if "anchor" not in params:
                errors.append(f"Line {line_num}: Missing required 'anchor' parameter")
                continue

            anchor_val = params["anchor"].lower()
            anchor_type: str | None = None
            anchor_block: int | None = None

            if anchor_val == "start":
                anchor_type = "start"
            elif anchor_val == "end":
                anchor_type = "end"
            elif anchor_val.startswith("before:"):
                anchor_type = "before"
                try:
                    n = int(anchor_val[len("before:"):])
                    if n < 1 or (total_blocks > 0 and n > total_blocks):
                        errors.append(f"Line {line_num}: Block N is out of range (1 to {max(1, total_blocks)}) in anchor='{params['anchor']}'")
                        continue
                    anchor_block = n
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid block number in '{params['anchor']}'")
                    continue
            elif anchor_val.startswith("after:"):
                anchor_type = "after"
                try:
                    n = int(anchor_val[len("after:"):])
                    if n < 1 or (total_blocks > 0 and n > total_blocks):
                        errors.append(f"Line {line_num}: Block N is out of range (1 to {max(1, total_blocks)}) in anchor='{params['anchor']}'")
                        continue
                    anchor_block = n
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid block number in '{params['anchor']}'")
                    continue
            else:
                errors.append(f"Line {line_num}: Unknown anchor '{params['anchor']}'")
                continue

            until_block: int | None = None
            if "until" in params:
                try:
                    m = int(params["until"])
                    if m < 1 or (total_blocks > 0 and m > total_blocks):
                        errors.append(f"Line {line_num}: Until block M is out of range (1 to {max(1, total_blocks)})")
                        continue
                    if anchor_block is not None and m < anchor_block:
                        errors.append(f"Line {line_num}: Until block M ({m}) must be greater than or equal to block N ({anchor_block})")
                        continue
                    until_block = m
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid until block number '{params['until']}'")
                    continue

            if "clip" not in params:
                errors.append(f"Line {line_num}: Missing required 'clip' parameter")
                continue
            clip_match = re.match(r"^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$", params["clip"])
            if not clip_match:
                errors.append(f"Line {line_num}: Invalid clip format '{params['clip']}'")
                continue
            clip_from = float(clip_match.group(1))
            clip_to = float(clip_match.group(2))
            if clip_from < 0 or clip_from > 30 or clip_to < 0 or clip_to > 30:
                errors.append(f"Line {line_num}: Clip part must be between 0 and 30 seconds")
                continue
            if clip_to <= clip_from:
                errors.append(f"Line {line_num}: Clip part To must be greater than From")
                continue

            if "volume" not in params:
                errors.append(f"Line {line_num}: Missing required 'volume' parameter")
                continue
            try:
                volume_db = float(params["volume"])
                if volume_db < -30 or volume_db > 0:
                    errors.append(f"Line {line_num}: Volume must be between -30 dB and 0 dB")
                    continue
            except ValueError:
                errors.append(f"Line {line_num}: Invalid volume value '{params['volume']}'")
                continue

            if "fade" not in params:
                errors.append(f"Line {line_num}: Missing required 'fade' parameter")
                continue
            fade_parts = params["fade"].split(",")
            if len(fade_parts) != 2:
                errors.append(f"Line {line_num}: Invalid fade format '{params['fade']}'")
                continue
            try:
                fade_in = float(fade_parts[0])
                fade_out = float(fade_parts[1])
                if fade_in < 0 or fade_in > 10 or fade_out < 0 or fade_out > 10:
                    errors.append(f"Line {line_num}: Fade values must be between 0 and 10 seconds")
                    continue
            except ValueError:
                errors.append(f"Line {line_num}: Invalid fade numbers in '{params['fade']}'")
                continue

            duck_val = params.get("duck", "").lower()
            if duck_val not in ("on", "off"):
                errors.append(f"Line {line_num}: Invalid duck value '{params.get('duck')}'; must be 'on' or 'off'")
                continue
            duck_speech = (duck_val == "on")

            music_directives.append({
                "line_num": line_num,
                "anchor_type": anchor_type,
                "anchor_block": anchor_block,
                "until_block": until_block,
                "clip_from": clip_from,
                "clip_to": clip_to,
                "volume_db": volume_db,
                "fade_in": fade_in,
                "fade_out": fade_out,
                "duck_speech": duck_speech,
                "raw_line": raw_line.strip(),
            })
            continue

        if stripped.startswith("#MUSIC_MODE:"):
            content = stripped[len("#MUSIC_MODE:"):].strip()
            tokens = content.split()
            params = {}
            for tok in tokens:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    params[k.lower()] = v
            if "blocks" not in params or "mode" not in params:
                errors.append(f"Line {line_num}: Invalid #MUSIC_MODE directive; requires blocks= and mode=")
                continue

            blocks_str = params["blocks"]
            if "-" in blocks_str:
                s_str, e_str = blocks_str.split("-", 1)
                try:
                    s_b = int(s_str)
                    e_b = int(e_str)
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid blocks range '{blocks_str}'")
                    continue
            else:
                try:
                    s_b = int(blocks_str)
                    e_b = s_b
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid block number '{blocks_str}'")
                    continue

            if s_b < 1 or (total_blocks > 0 and e_b > total_blocks):
                errors.append(f"Line {line_num}: Block range {blocks_str} is out of range (1 to {max(1, total_blocks)})")
                continue
            if e_b < s_b:
                errors.append(f"Line {line_num}: End block ({e_b}) must be greater than or equal to start block ({s_b})")
                continue

            mode = params["mode"].lower()
            if mode not in ("normal", "duck", "silent"):
                errors.append(f"Line {line_num}: Invalid mode '{params['mode']}'; must be 'normal', 'duck', or 'silent'")
                continue

            music_modes.append({
                "line_num": line_num,
                "start_block": s_b,
                "end_block": e_b,
                "mode": mode,
                "raw_line": raw_line.strip(),
            })
            continue

    return {
        "music_prompt": music_prompt,
        "music_directives": music_directives,
        "music_modes": music_modes,
        "errors": errors,
    }


def resolve_placement_span(dir_item: dict[str, Any], total_blocks: int) -> dict[str, Any]:
    """Resolves block span and description for a #MUSIC placement."""
    a_type = dir_item["anchor_type"]
    until = dir_item.get("until_block")
    a_block = dir_item.get("anchor_block", 1)

    if a_type == "start":
        end_b = until or total_blocks
        desc = f"Plays from start of podcast until block {until}" if until else "Plays at start of podcast"
        return {"start": 1, "end": end_b, "desc": desc}
    elif a_type == "end":
        return {"start": total_blocks, "end": total_blocks, "desc": "Plays at end of podcast"}
    elif a_type == "before":
        if until:
            return {"start": a_block, "end": until, "desc": f"Plays from block {a_block} to block {until}"}
        return {"start": a_block, "end": a_block, "desc": f"Plays before block {a_block}"}
    elif a_type == "after":
        if until:
            return {"start": a_block, "end": until, "desc": f"Plays after block {a_block} until block {until}"}
        return {"start": a_block, "end": min(total_blocks, a_block + 1), "desc": f"Plays after block {a_block}"}
    return {"start": 1, "end": total_blocks, "desc": "Unknown anchor"}


def parse_script_file(file_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parses script blocks and directives between START_MARKER and END_MARKER."""
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: Script file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks: list[dict[str, Any]] = []
    has_start = any(line.strip() == START_MARKER for line in lines)
    has_end = any(line.strip() == END_MARKER for line in lines)
    use_markers = has_start and has_end

    inside_block = False

    for line_num, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if use_markers:
            if stripped == START_MARKER:
                inside_block = True
                continue
            if stripped == END_MARKER:
                inside_block = False
                break
            if not inside_block:
                continue

        if not stripped or stripped.startswith("#"):
            continue

        first_bar = raw_line.find("|")
        second_bar = raw_line.find("|", first_bar + 1)
        if first_bar == -1 or second_bar == -1:
            print(
                f"Error: Line {line_num} does not have exactly three parts separated by '|':\n  {raw_line.strip()}",
                file=sys.stderr,
            )
            sys.exit(1)

        speaker = raw_line[:first_bar].strip()
        style = raw_line[first_bar + 1 : second_bar].strip()
        text = raw_line[second_bar + 1 :].strip()

        if speaker not in SPEAKER_MAP:
            print(
                f"Error: Line {line_num} has invalid speaker '{speaker}'. Must be 'Rami' or 'Leda'.",
                file=sys.stderr,
            )
            sys.exit(1)

        blocks.append(
            {
                "line_num": line_num,
                "speaker": speaker,
                "api_speaker": SPEAKER_MAP[speaker],
                "style": style,
                "text": text,
            }
        )

    directives_info = parse_script_directives(lines, len(blocks))
    if directives_info["errors"]:
        for err in directives_info["errors"]:
            print(f"Directive Error: {err}", file=sys.stderr)
        sys.exit(1)

    if not blocks:
        print(
            f"Error: No valid script lines found in '{file_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return blocks, directives_info


def strip_bracket_tags(text: str) -> str:
    """Removes bracket tags like [excited], [calm] from text."""
    return re.sub(r"\[[^\]]*\]", "", text)


def count_words(text: str) -> int:
    """Counts words after removing bracket tags."""
    cleaned = strip_bracket_tags(text).strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """Reads bits_per_sample and rate from an audio MIME type string."""
    bits_per_sample = 16
    rate = 24000

    if not mime_type:
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    parts = mime_type.split(";")
    for param in parts:
        param_stripped = param.strip()
        lower_param = param_stripped.lower()
        if lower_param.startswith("rate="):
            try:
                rate_str = param_stripped.split("=", 1)[1].strip()
                parsed_rate = int(rate_str)
                if parsed_rate > 0:
                    rate = parsed_rate
            except (ValueError, IndexError):
                pass
        elif lower_param.startswith("audio/l"):
            try:
                bit_str = lower_param[len("audio/l") :]
                parsed_bits = int(bit_str)
                if parsed_bits > 0:
                    bits_per_sample = parsed_bits
            except ValueError:
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def extract_pcm_payload(raw_buffer: bytes) -> bytes:
    """Extracts raw PCM data if raw_buffer is already a RIFF/WAVE container."""
    if len(raw_buffer) >= 44 and raw_buffer[:4] == b"RIFF" and raw_buffer[8:12] == b"WAVE":
        offset = 12
        while offset + 8 <= len(raw_buffer):
            subchunk_id = raw_buffer[offset : offset + 4]
            subchunk_size = struct.unpack("<I", raw_buffer[offset + 4 : offset + 8])[0]
            if subchunk_id == b"data":
                start = offset + 8
                end = min(start + subchunk_size, len(raw_buffer))
                return raw_buffer[start:end]
            offset += 8 + subchunk_size
        return raw_buffer[44:]
    return raw_buffer


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Creates a standard 44-byte RIFF WAV header and appends raw PCM bytes."""
    meta = parse_audio_mime_type(mime_type)
    rate = meta["rate"]
    bits_per_sample = meta["bits_per_sample"]
    channels = 1

    byte_rate = rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_length = len(audio_data)
    chunk_size = 36 + data_length

    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, chunk_size)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)
    struct.pack_into("<H", header, 20, 1)  # PCM
    struct.pack_into("<H", header, 22, channels)
    struct.pack_into("<I", header, 24, rate)
    struct.pack_into("<I", header, 28, byte_rate)
    struct.pack_into("<H", header, 32, block_align)
    struct.pack_into("<H", header, 34, bits_per_sample)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, data_length)

    return bytes(header) + audio_data


def build_speech_config(voice1: str = "Rami", voice2: str = "Leda") -> types.SpeechConfig:
    """Builds two-speaker conversational speech configuration."""
    return types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            mode="CONVERSATIONAL",
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker="Speaker 1",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice1,
                        )
                    ),
                ),
                types.SpeakerVoiceConfig(
                    speaker="Speaker 2",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice2,
                        )
                    ),
                ),
            ],
        )
    )


def build_chunk_parts(chunk_blocks: list[dict[str, Any]]) -> list[types.Part]:
    """Builds Part items for one batch of dialogue blocks."""
    parts: list[types.Part] = []

    for block in chunk_blocks:
        api_speaker = block["api_speaker"]
        exact_text = block["text"]

        if hasattr(types, "SpeechMetadata"):
            parts.append(
                types.Part(
                    text=exact_text,
                    speech_metadata=types.SpeechMetadata(
                        speaker=api_speaker,
                    ),
                )
            )
        else:
            parts.append(types.Part.from_text(text=f"{api_speaker}: {exact_text}"))

    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Podcast Audio Generator CLI")
    parser.add_argument(
        "--lang",
        choices=["en", "fa", "ar"],
        default="en",
        help="Language script to process (en, fa, ar). Default is en.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TTS_MODEL,
        help=f"TTS Model name to use for synthesis. Default is {DEFAULT_TTS_MODEL}.",
    )
    parser.add_argument(
        "--voice1",
        default=None,
        help="Voice name for Speaker 1 (default: Rami).",
    )
    parser.add_argument(
        "--voice2",
        default=None,
        help="Voice name for Speaker 2 (default: Leda).",
    )
    parser.add_argument(
        "--name1",
        default="Arash",
        help="Display name for role 1 (default: Arash).",
    )
    parser.add_argument(
        "--name2",
        default="Sara",
        help="Display name for role 2 (default: Sara).",
    )
    args = parser.parse_args()
    lang = args.lang
    name1 = args.name1 or "Arash"
    name2 = args.name2 or "Sara"
    selected_model = args.model or DEFAULT_TTS_MODEL

    default_for_model = DEFAULT_VOICES.get(selected_model, {"voice1": "Rami", "voice2": "Leda"})
    voice1 = args.voice1 or default_for_model["voice1"]
    voice2 = args.voice2 or default_for_model["voice2"]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    input_file = f"script.{lang}.txt"
    blocks, directives_info = parse_script_file(input_file)
    total_blocks = len(blocks)
    total_chunks = (total_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE

    min_required = MIN_BLOCKS_BY_LANG.get(lang, 1)
    if total_blocks < min_required:
        print(
            f"Error: Script file '{input_file}' contains {total_blocks} blocks, "
            f"which is below the required {min_required} blocks for language '{lang}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if lang == "en" and total_blocks != EXPECTED_BLOCKS_BY_LANG["en"]:
        print(
            f"Warning: Script '{input_file}' contains {total_blocks} blocks "
            f"(expected {EXPECTED_BLOCKS_BY_LANG['en']} blocks)."
        )

    rami_count = sum(1 for b in blocks if b["speaker"] == "Rami")
    leda_count = sum(1 for b in blocks if b["speaker"] == "Leda")
    print(f"Speakers: {name1} (Speaker 1 — {voice1}), {name2} (Speaker 2 — {voice2})")
    print(f"Total blocks: {total_blocks} (Rami/{name1}: {rami_count}, Leda/{name2}: {leda_count})")
    print(f"Parsed {total_blocks} dialogue blocks from '{input_file}'.")

    # Report directives if present
    if directives_info["music_prompt"] or directives_info["music_directives"] or directives_info["music_modes"]:
        print(f"\n--- Music Directives ({len(directives_info['music_directives'])} placements, {len(directives_info['music_modes'])} modes) ---")
        if directives_info["music_prompt"]:
            print(f"Music Prompt: \"{directives_info['music_prompt']}\"")
        for p_idx, p in enumerate(directives_info["music_directives"], start=1):
            span = resolve_placement_span(p, total_blocks)
            print(
                f"  Placement {p_idx}: anchor={p['anchor_type']} "
                f"(anchorBlock={p['anchor_block']}, until={p['until_block']}) "
                f"clip={p['clip_from']}-{p['clip_to']}s vol={p['volume_db']}dB "
                f"fade={p['fade_in']},{p['fade_out']}s duck={'on' if p['duck_speech'] else 'off'} "
                f"-> {span['desc']} (blocks {span['start']}-{span['end']})"
            )
        for m_idx, m in enumerate(directives_info["music_modes"], start=1):
            print(f"  Music Mode {m_idx}: blocks={m['start_block']}-{m['end_block']} mode={m['mode']}")
        print("--------------------------------------------------------------------\n")
    for idx, b in enumerate(blocks, start=1):
        block_word_count = count_words(b["text"])
        if block_word_count > 60:
            print(
                f"Warning: Block {idx} contains {block_word_count} words (> 60 words). "
                f"Very long blocks may be truncated by the model."
            )
        print(f"  [Block {idx}] {b['speaker']} ({b['api_speaker']}) | Style: '{b['style']}'")

    if genai is None or types is None:
        print("Error: The 'google-genai' package is required. Install it with: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    speech_config = build_speech_config(voice1=voice1, voice2=voice2)

    config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=speech_config,
    )

    collected_pcm_chunks: list[bytes] = []
    chunk_durations: list[float] = []
    total_expected_seconds = 0.0
    total_actual_seconds = 0.0
    detected_mime_type = "audio/L16;rate=24000"

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_blocks)
        chunk_blocks = blocks[start_idx:end_idx]

        first_block_num = start_idx + 1
        last_block_num = end_idx

        # Calculate expected seconds: words after bracket tags / 150 * 60
        chunk_words = sum(count_words(b["text"]) for b in chunk_blocks)
        expected_seconds = (chunk_words / 150.0) * 60.0 if chunk_words > 0 else 1.0
        total_expected_seconds += expected_seconds

        print(
            f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} "
            f"(blocks {first_block_num} to {last_block_num}, {chunk_words} words, ~{expected_seconds:.1f}s expected)..."
        )
        for b_idx, block in enumerate(chunk_blocks, start=first_block_num):
            cleaned_debug_text = strip_bracket_tags(block['text']).strip()
            print(
                f"  Debug Block {b_idx}: Speaker={block['speaker']} "
                f"Style='{block['style']}' Text='{cleaned_debug_text[:40]}...'"
            )

        parts = build_chunk_parts(chunk_blocks)
        contents = [types.Content(role="user", parts=parts)]

        chunk_audio_bytes: bytes | None = None
        last_error: Exception | str | None = None
        final_coverage: float = 0.0
        final_actual_seconds: float = 0.0

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=contents,
                    config=config,
                )

                extracted = bytearray()
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and getattr(inline_data, "data", None):
                                    raw_data = inline_data.data
                                    if isinstance(raw_data, str):
                                        raw_data = base64.b64decode(raw_data)
                                    if getattr(inline_data, "mime_type", None):
                                        detected_mime_type = inline_data.mime_type
                                    extracted.extend(extract_pcm_payload(raw_data))

                if not extracted:
                    raise RuntimeError("API response contained no audio bytes.")

                candidate_pcm = bytes(extracted)
                sample_rate = parse_audio_mime_type(detected_mime_type)["rate"]
                actual_seconds = len(candidate_pcm) / (sample_rate * 2)
                coverage_ratio = actual_seconds / expected_seconds

                if coverage_ratio < 0.6:
                    raise RuntimeError(
                        f"Coverage too low: {coverage_ratio:.2f} < 0.60 "
                        f"(expected {expected_seconds:.1f}s, got {actual_seconds:.1f}s)"
                    )

                chunk_audio_bytes = candidate_pcm
                final_coverage = coverage_ratio
                final_actual_seconds = actual_seconds
                break
            except Exception as exc:
                last_error = exc
                if attempt <= MAX_RETRIES:
                    print(
                        f"Chunk {chunk_idx + 1}/{total_chunks} (blocks {first_block_num}-{last_block_num}) "
                        f"failed (attempt {attempt}/{MAX_RETRIES}): {exc}. "
                        f"Retrying in {RETRY_WAIT_SECONDS} seconds..."
                    )
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    print(
                        f"Error: Chunk {chunk_idx + 1}/{total_chunks} covering blocks "
                        f"{first_block_num} to {last_block_num} failed after {MAX_RETRIES} retries. "
                        f"Last error: {last_error}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        if not chunk_audio_bytes:
            print(
                f"Error: Chunk {chunk_idx + 1}/{total_chunks} (blocks {first_block_num}-{last_block_num}) "
                f"produced no valid audio.",
                file=sys.stderr,
            )
            sys.exit(1)

        collected_pcm_chunks.append(chunk_audio_bytes)
        chunk_durations.append(final_actual_seconds)
        total_actual_seconds += final_actual_seconds

        print(
            f"Chunk {chunk_idx + 1}/{total_chunks} coverage: {final_coverage:.2f} "
            f"(expected {int(round(expected_seconds))}s, got {int(round(final_actual_seconds))}s)"
        )
        print(
            f"Progress: Completed chunk {chunk_idx + 1}/{total_chunks} "
            f"(blocks {first_block_num}-{last_block_num}, {len(chunk_audio_bytes)} bytes PCM)."
        )

    total_coverage = (
        total_actual_seconds / total_expected_seconds
        if total_expected_seconds > 0
        else 1.0
    )
    print(
        f"Total coverage: {total_coverage:.2f} "
        f"(expected {total_expected_seconds:.0f}s, got {total_actual_seconds:.0f}s)"
    )

    if total_coverage < 0.85:
        print(
            f"Error: Total coverage {total_coverage:.2f} is below 0.85 threshold. "
            f"Audio is too short for the script (expected {total_expected_seconds:.0f}s, got {total_actual_seconds:.0f}s).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nFinal audio duration: {total_actual_seconds:.1f} seconds across {total_chunks} chunk(s).")

    all_pcm_bytes = b"".join(collected_pcm_chunks)
    wav_bytes = convert_to_wav(all_pcm_bytes, detected_mime_type)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = outputs_dir / f"podcast_{lang}_{timestamp}.wav"

    with output_path.open("wb") as out_f:
        out_f.write(wav_bytes)

    sample_rate = parse_audio_mime_type(detected_mime_type)["rate"]
    pcm_duration = len(all_pcm_bytes) / (sample_rate * 2)

    print(
        f"Successfully generated '{output_path}' ({len(wav_bytes)} bytes) "
        f"for {total_blocks} blocks across {total_chunks} chunk(s)."
    )
    print(
        f"Output file: {output_path}, Size: {len(wav_bytes)} bytes, "
        f"Actual duration: {pcm_duration:.2f} seconds."
    )


if __name__ == "__main__":
    main()
