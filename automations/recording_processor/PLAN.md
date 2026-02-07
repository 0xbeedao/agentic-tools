# Recording Processor CLI Plan

## Goal
Create a Python CLI that processes audio/video recordings into structured text outputs using `faster-whisper`.

## Assumptions
- Inputs are audio or video files (single file or directory batch).
- Outputs include a plain transcript and a timestamped format.
- The tool is used from the command line and is scriptable.

## Non-goals
- No web UI.
- No real-time streaming transcription.
- No speaker diarization.

## Inputs
- File path(s): audio/video.
- Optional: output directory.

## Outputs (File-Friendly Chain)
- `segments.jsonl` (one segment per line, timestamped).
- `chunks.jsonl` (segments grouped into summarization-friendly chunks).
- `manifest.json` (input metadata, model/lang, chunking params, hashes).
- Optional: `transcript.txt` (full text) and `transcript.srt` (subtitles).

### Bundle File Shapes
`segments.jsonl` line example:
```json
{"id":"seg-0003","start":12.34,"end":18.02,"text":"...","source":"input.wav","model":"base","lang":"en"}
```

`chunks.jsonl` line example:
```json
{"id":"chunk-0001","start":0.0,"end":58.2,"token_count":982,"text":"...","segment_ids":["seg-0001","seg-0002","seg-0003"]}
```

`manifest.json` example:
```json
{
  "source_path": "./sample/input.wav",
  "source_hash": "sha256:...",
  "created_at": "2026-01-29T00:00:00Z",
  "model": "base",
  "language": "en",
  "device": "auto",
  "chunk_tokens": 1000,
  "chunk_overlap": 100,
  "outputs": {
    "segments": "segments.jsonl",
    "chunks": "chunks.jsonl",
    "transcript": "transcript.txt",
    "srt": "transcript.srt"
  }
}
```

## CLI Design

### Command 1: `recording-transcribe`
Transcribe audio/video and generate chunked outputs.

Arguments:
- `--input <path>` (file or directory)
- `--output <dir>` (default: alongside input)
- `--whisper-model <name>` (default: `base`)
- `--language <code>` (optional)
- `--device <cpu|cuda|auto>` (default: `auto`)
- `--emit <bundle|txt|srt|all>` (default: `bundle`)
- `--chunk-tokens <int>` (default: `1000`)
- `--chunk-overlap <int>` (default: `100`)
- `--overwrite` (allow overwriting outputs)
- `--batch-mode <sequential|parallel>` (default: `sequential`)
- `--config <file>` (load CLI args from config file)
- `--check-ffmpeg` (early validation with setup guidance)

### Command 2: `recording-categorize`
Categorize chunks into daily summary document.

Arguments:
- `--input <dir>` (directory with chunks.jsonl from transcribe)
- `--output <path>` (default: daily-summary-YYYY-MM-DD.md)
- `--categories <json>` (default: `["Project Notes","Ideas","Todos","Journal"]`)
- `--model <id>` (default: `openrouter/x-ai/grok-4-fast`)
- `--dedupe` (hash-based deduplication to skip processed chunks)
- `--progress` (show progress indicator for long runs)

## Processing Flow

### `recording-transcribe`
1. Validate input path(s) and output directory.
2. Check ffmpeg availability with actionable setup guidance.
3. If video, extract audio (ffmpeg) to a temp file.
4. Compute source hash and check existing outputs for dedup (skip if unchanged).
5. Run `faster-whisper` transcription with selected model/device, showing progress % and ETA.
6. Write `segments.jsonl` and `manifest.json`.
7. Chunk segments and write `chunks.jsonl` (validate max chunk size).
8. Write optional text/subtitle outputs.
9. Clean up temp files.
10. Batch mode: process `sequential` or `parallel` with timeout.

### `recording-categorize`
1. Read `chunks.jsonl` from input directory.
2. Filter out already-processed chunks using dedupe tracking.
3. Send each chunk to LLM for classification (default: `openrouter/x-ai/grok-4-fast`).
4. Append categorized content to daily summary doc under headings.
5. Write dedupe tracking for idempotency on subsequent runs.

## Error Handling
- Clear error when input is missing or unsupported.
- Actionable guidance for missing ffmpeg or CUDA libs.
- Non-zero exit codes for failures.

## Dependencies
- `faster-whisper`
- `ffmpeg` system package
- `typer` or `argparse` for CLI
- `llm` (https://github.com/simonw/llm) for categorization

## Testing
- Unit tests for path validation and output formatting.
- Golden test for a short sample audio file.
- Sample recordings live in `./sample` with expected outputs.
- Integration tests for both commands end-to-end.

## Decisions
- No custom metadata or formatting required now.
- Batch mode does not need to preserve subdirectory structure.
- Separated transcription and categorization into distinct commands for modularity.
- Configurable daily summary format (default: markdown with date).
- Hash-based deduplication prevents re-processing unchanged inputs.
