// Directive parsing, types, and block anchor resolution for background music
// Shared rules between UI browser code and make_podcast.py

export type MusicModeType = "normal" | "duck" | "silent";

export interface ParsedMusicDirective {
  lineNum: number;
  anchorType: "start" | "end" | "before" | "after";
  anchorBlock?: number;
  untilBlock?: number;
  clipFrom: number;
  clipTo: number;
  volumeDb: number;
  fadeIn: number;
  fadeOut: number;
  duckSpeech: boolean;
  rawLine: string;
}

export interface ParsedMusicModeDirective {
  lineNum: number;
  startBlock: number;
  endBlock: number;
  mode: MusicModeType;
  rawLine: string;
}

export interface ParsedDirectivesResult {
  musicPrompt: string | null;
  musicDirectives: ParsedMusicDirective[];
  musicModeDirectives: ParsedMusicModeDirective[];
  errors: { lineNum: number; problem: string; preview: string }[];
}

export interface ResolvedBlockSpan {
  startBlockSpan: number; // 1-based block index or 1
  endBlockSpan: number;   // 1-based block index or totalBlocks
  description: string;
}

/**
 * Parses # directives from raw script lines.
 * Supported:
 *   #MUSIC_PROMPT: <text> (one allowed)
 *   #MUSIC: anchor=<start|end|before:N|after:N> [until=M] clip=<from>-<to> volume=<dB> fade=<in>,<out> duck=<on|off>
 *   #MUSIC_MODE: blocks=<N> mode=<normal|duck|silent> or blocks=<N>-<M> mode=<...>
 * Reject "at=" or "sec=" with "use block anchors".
 */
export function parseScriptDirectives(
  rawContent: string,
  totalBlocks: number
): ParsedDirectivesResult {
  const lines = rawContent.split(/\r?\n/);
  const result: ParsedDirectivesResult = {
    musicPrompt: null,
    musicDirectives: [],
    musicModeDirectives: [],
    errors: [],
  };

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

    if (!stripped || !stripped.startsWith("#")) {
      continue;
    }

    // 1. #MUSIC_PROMPT:
    if (stripped.startsWith("#MUSIC_PROMPT:")) {
      const promptText = stripped.slice("#MUSIC_PROMPT:".length).trim();
      if (!promptText) {
        result.errors.push({
          lineNum,
          problem: "MUSIC_PROMPT directive has empty prompt text",
          preview: rawLine,
        });
      } else if (result.musicPrompt !== null) {
        result.errors.push({
          lineNum,
          problem: "Only one #MUSIC_PROMPT directive is allowed",
          preview: rawLine,
        });
      } else {
        result.musicPrompt = promptText;
      }
      continue;
    }

    // 2. #MUSIC:
    if (stripped.startsWith("#MUSIC:")) {
      const content = stripped.slice("#MUSIC:".length).trim();

      // Check for forbidden time fields: at= or sec=
      if (/\b(at|sec)=/i.test(content)) {
        result.errors.push({
          lineNum,
          problem: "Time-based placement is not supported; use block anchors",
          preview: rawLine,
        });
        continue;
      }

      // Parse key=value pairs
      const tokens = content.split(/\s+/).filter(Boolean);
      const params: Record<string, string> = {};
      let parseFailed = false;

      for (const tok of tokens) {
        const eqIdx = tok.indexOf("=");
        if (eqIdx === -1) {
          result.errors.push({
            lineNum,
            problem: `Invalid parameter format '${tok}'; expected key=value`,
            preview: rawLine,
          });
          parseFailed = true;
          break;
        }
        const k = tok.slice(0, eqIdx).toLowerCase();
        const v = tok.slice(eqIdx + 1);
        params[k] = v;
      }

      if (parseFailed) continue;

      // Validate anchor
      if (!params.anchor) {
        result.errors.push({
          lineNum,
          problem: "Missing required 'anchor' parameter",
          preview: rawLine,
        });
        continue;
      }

      let anchorType: "start" | "end" | "before" | "after";
      let anchorBlock: number | undefined;

      const anchorVal = params.anchor.toLowerCase();
      if (anchorVal === "start") {
        anchorType = "start";
      } else if (anchorVal === "end") {
        anchorType = "end";
      } else if (anchorVal.startsWith("before:")) {
        anchorType = "before";
        const nStr = anchorVal.slice("before:".length);
        const n = parseInt(nStr, 10);
        if (isNaN(n) || n < 1 || (totalBlocks > 0 && n > totalBlocks)) {
          result.errors.push({
            lineNum,
            problem: `Block N is out of range (1 to ${Math.max(1, totalBlocks)}) in anchor='${params.anchor}'`,
            preview: rawLine,
          });
          continue;
        }
        anchorBlock = n;
      } else if (anchorVal.startsWith("after:")) {
        anchorType = "after";
        const nStr = anchorVal.slice("after:".length);
        const n = parseInt(nStr, 10);
        if (isNaN(n) || n < 1 || (totalBlocks > 0 && n > totalBlocks)) {
          result.errors.push({
            lineNum,
            problem: `Block N is out of range (1 to ${Math.max(1, totalBlocks)}) in anchor='${params.anchor}'`,
            preview: rawLine,
          });
          continue;
        }
        anchorBlock = n;
      } else {
        result.errors.push({
          lineNum,
          problem: `Unknown anchor '${params.anchor}'; must be start, end, before:N, or after:N`,
          preview: rawLine,
        });
        continue;
      }

      // Optional until
      let untilBlock: number | undefined;
      if (params.until !== undefined) {
        const m = parseInt(params.until, 10);
        if (isNaN(m) || m < 1 || (totalBlocks > 0 && m > totalBlocks)) {
          result.errors.push({
            lineNum,
            problem: `Until block M is out of range (1 to ${Math.max(1, totalBlocks)})`,
            preview: rawLine,
          });
          continue;
        }
        if (anchorBlock !== undefined && m < anchorBlock) {
          result.errors.push({
            lineNum,
            problem: `Until block M (${m}) must be greater than or equal to block N (${anchorBlock})`,
            preview: rawLine,
          });
          continue;
        }
        untilBlock = m;
      }

      // clip=<from>-<to>
      if (!params.clip) {
        result.errors.push({
          lineNum,
          problem: "Missing required 'clip' parameter (e.g. clip=0-10)",
          preview: rawLine,
        });
        continue;
      }
      const clipMatch = params.clip.match(/^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$/);
      if (!clipMatch) {
        result.errors.push({
          lineNum,
          problem: `Invalid clip format '${params.clip}'; expected from-to (e.g. clip=0-10)`,
          preview: rawLine,
        });
        continue;
      }
      const clipFrom = parseFloat(clipMatch[1]);
      const clipTo = parseFloat(clipMatch[2]);
      if (clipFrom < 0 || clipFrom > 30 || clipTo < 0 || clipTo > 30) {
        result.errors.push({
          lineNum,
          problem: "Clip part must be between 0 and 30 seconds",
          preview: rawLine,
        });
        continue;
      }
      if (clipTo <= clipFrom) {
        result.errors.push({
          lineNum,
          problem: "Clip part To must be greater than From",
          preview: rawLine,
        });
        continue;
      }

      // volume=<dB> (-30 to 0)
      if (params.volume === undefined) {
        result.errors.push({
          lineNum,
          problem: "Missing required 'volume' parameter (e.g. volume=-18)",
          preview: rawLine,
        });
        continue;
      }
      const volumeDb = parseFloat(params.volume);
      if (isNaN(volumeDb) || volumeDb < -30 || volumeDb > 0) {
        result.errors.push({
          lineNum,
          problem: `Volume must be between -30 dB and 0 dB (got '${params.volume}')`,
          preview: rawLine,
        });
        continue;
      }

      // fade=<in>,<out> (0 to 10 each)
      if (!params.fade) {
        result.errors.push({
          lineNum,
          problem: "Missing required 'fade' parameter (e.g. fade=1.5,2)",
          preview: rawLine,
        });
        continue;
      }
      const fadeParts = params.fade.split(",");
      if (fadeParts.length !== 2) {
        result.errors.push({
          lineNum,
          problem: `Invalid fade format '${params.fade}'; expected in,out (e.g. fade=1.5,2)`,
          preview: rawLine,
        });
        continue;
      }
      const fadeIn = parseFloat(fadeParts[0]);
      const fadeOut = parseFloat(fadeParts[1]);
      if (isNaN(fadeIn) || fadeIn < 0 || fadeIn > 10 || isNaN(fadeOut) || fadeOut < 0 || fadeOut > 10) {
        result.errors.push({
          lineNum,
          problem: `Fade values must be between 0 and 10 seconds (got '${params.fade}')`,
          preview: rawLine,
        });
        continue;
      }

      // duck=<on|off>
      if (!params.duck || (params.duck.toLowerCase() !== "on" && params.duck.toLowerCase() !== "off")) {
        result.errors.push({
          lineNum,
          problem: `Invalid duck value '${params.duck}'; must be 'on' or 'off'`,
          preview: rawLine,
        });
        continue;
      }
      const duckSpeech = params.duck.toLowerCase() === "on";

      result.musicDirectives.push({
        lineNum,
        anchorType,
        anchorBlock,
        untilBlock,
        clipFrom,
        clipTo,
        volumeDb,
        fadeIn,
        fadeOut,
        duckSpeech,
        rawLine,
      });
      continue;
    }

    // 3. #MUSIC_MODE:
    if (stripped.startsWith("#MUSIC_MODE:")) {
      const content = stripped.slice("#MUSIC_MODE:".length).trim();
      const tokens = content.split(/\s+/).filter(Boolean);
      const params: Record<string, string> = {};

      for (const tok of tokens) {
        const eqIdx = tok.indexOf("=");
        if (eqIdx !== -1) {
          params[tok.slice(0, eqIdx).toLowerCase()] = tok.slice(eqIdx + 1);
        }
      }

      if (!params.blocks || !params.mode) {
        result.errors.push({
          lineNum,
          problem: "Invalid #MUSIC_MODE directive; requires blocks=<N|N-M> and mode=<normal|duck|silent>",
          preview: rawLine,
        });
        continue;
      }

      let startBlock: number;
      let endBlock: number;

      if (params.blocks.includes("-")) {
        const [sStr, eStr] = params.blocks.split("-");
        startBlock = parseInt(sStr, 10);
        endBlock = parseInt(eStr, 10);
      } else {
        startBlock = parseInt(params.blocks, 10);
        endBlock = startBlock;
      }

      if (
        isNaN(startBlock) ||
        isNaN(endBlock) ||
        startBlock < 1 ||
        (totalBlocks > 0 && endBlock > totalBlocks)
      ) {
        result.errors.push({
          lineNum,
          problem: `Block range ${params.blocks} is out of range (1 to ${Math.max(1, totalBlocks)})`,
          preview: rawLine,
        });
        continue;
      }

      if (endBlock < startBlock) {
        result.errors.push({
          lineNum,
          problem: `End block (${endBlock}) must be greater than or equal to start block (${startBlock})`,
          preview: rawLine,
        });
        continue;
      }

      const modeLower = params.mode.toLowerCase();
      if (modeLower !== "normal" && modeLower !== "duck" && modeLower !== "silent") {
        result.errors.push({
          lineNum,
          problem: `Invalid mode '${params.mode}'; must be 'normal', 'duck', or 'silent'`,
          preview: rawLine,
        });
        continue;
      }

      result.musicModeDirectives.push({
        lineNum,
        startBlock,
        endBlock,
        mode: modeLower as MusicModeType,
        rawLine,
      });
      continue;
    }

    // Any other # line is a plain comment, ignored.
  }

  return result;
}

/**
 * Resolves the span text and block boundaries for a parsed #MUSIC directive.
 */
export function resolvePlacementSpan(
  dir: ParsedMusicDirective,
  totalBlocks: number
): ResolvedBlockSpan {
  if (dir.anchorType === "start") {
    const end = dir.untilBlock || totalBlocks;
    return {
      startBlockSpan: 1,
      endBlockSpan: end,
      description: dir.untilBlock
        ? `Plays from start of podcast until block ${dir.untilBlock}`
        : "Plays at start of podcast",
    };
  }
  if (dir.anchorType === "end") {
    return {
      startBlockSpan: totalBlocks,
      endBlockSpan: totalBlocks,
      description: "Plays at end of podcast",
    };
  }
  if (dir.anchorType === "before") {
    const n = dir.anchorBlock || 1;
    if (dir.untilBlock) {
      return {
        startBlockSpan: n,
        endBlockSpan: dir.untilBlock,
        description: `Plays from block ${n} to block ${dir.untilBlock}`,
      };
    }
    return {
      startBlockSpan: n,
      endBlockSpan: n,
      description: `Plays before block ${n}`,
    };
  }
  if (dir.anchorType === "after") {
    const n = dir.anchorBlock || 1;
    if (dir.untilBlock) {
      return {
        startBlockSpan: n,
        endBlockSpan: dir.untilBlock,
        description: `Plays after block ${n} until block ${dir.untilBlock}`,
      };
    }
    return {
      startBlockSpan: n,
      endBlockSpan: Math.min(totalBlocks, n + 1),
      description: `Plays after block ${n}`,
    };
  }

  return {
    startBlockSpan: 1,
    endBlockSpan: totalBlocks,
    description: "Unknown anchor",
  };
}
