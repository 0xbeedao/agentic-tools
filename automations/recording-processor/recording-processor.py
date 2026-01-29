import datetime
import hashlib
import json
import shutil
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import click

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    elif path.suffix in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+")
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    else:
        raise ValueError("Unsupported config format. Use .json or .toml")

    if "recording_transcribe" in config and isinstance(
        config["recording_transcribe"], dict
    ):
        config = config["recording_transcribe"]

    normalized = {}
    for key, value in config.items():
        normalized[key.replace("-", "_")] = value

    return normalized


@click.group()
def cli():
    pass


@click.command("transcribe")
@click.option("--input", default=None, help="Audio/video file or directory")
@click.option("--output", default=None, help="Output directory")
@click.option("--whisper-model", default="base", help="Whisper model to use")
@click.option("--language", default=None, help="Language code (e.g., en, es)")
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["cpu", "cuda", "auto"]),
    help="Device to use for inference",
)
@click.option(
    "--emit",
    default="bundle",
    type=click.Choice(["bundle", "txt", "srt", "all"]),
    help="Output format to emit",
)
@click.option("--chunk-tokens", default=1000, type=int, help="Target tokens per chunk")
@click.option(
    "--chunk-overlap", default=100, type=int, help="Overlap tokens between chunks"
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output files"
)
@click.option(
    "--batch-mode",
    default="sequential",
    type=click.Choice(["sequential", "parallel"]),
    help="Batch processing mode",
)
@click.option("--config", default=None, help="Path to JSON or TOML config file")
@click.option(
    "--check-ffmpeg", is_flag=True, default=False, help="Check if ffmpeg is available"
)
def transcribe(
    input,
    output,
    whisper_model,
    language,
    device,
    emit,
    chunk_tokens,
    chunk_overlap,
    overwrite,
    batch_mode,
    config,
    check_ffmpeg,
):
    defaults = {}
    if config:
        defaults = _load_config(Path(config))

    input = input or defaults.get("input")
    output = output or defaults.get("output")
    whisper_model = whisper_model or defaults.get("whisper_model", "base")
    language = language or defaults.get("language")
    device = device or defaults.get("device", "auto")
    emit = emit or defaults.get("emit", "bundle")
    chunk_tokens = chunk_tokens or defaults.get("chunk_tokens", 1000)
    chunk_overlap = chunk_overlap or defaults.get("chunk_overlap", 100)
    overwrite = overwrite or defaults.get("overwrite", False)
    batch_mode = batch_mode or defaults.get("batch_mode", "sequential")
    check_ffmpeg = check_ffmpeg or defaults.get("check_ffmpeg", False)

    if not input:
        click.echo("Error: --input is required (or provide it via --config).", err=True)
        raise click.Abort()

    class Args:
        def __init__(self):
            self.input = input
            self.output = output
            self.whisper_model = whisper_model
            self.language = language
            self.device = device
            self.emit = emit
            self.chunk_tokens = chunk_tokens
            self.chunk_overlap = chunk_overlap
            self.overwrite = overwrite
            self.batch_mode = batch_mode
            self.check_ffmpeg = check_ffmpeg

    args = Args()
    result = run_recording_transcribe(args)
    raise SystemExit(result)


@click.command("categorize")
@click.option("--input", required=True, help="Directory containing chunks.jsonl")
@click.option(
    "--output",
    default=None,
    help="Output markdown file (default: daily-summary-YYYY-MM-DD.md)",
)
@click.option(
    "--categories",
    default='["Project Notes","Ideas","Todos","Journal"]',
    help="JSON array of category names",
)
@click.option(
    "--model",
    default="openrouter/x-ai/grok-4-fast",
    help="LLM model ID for categorization",
)
@click.option(
    "--dedupe",
    is_flag=True,
    default=False,
    help="Skip chunks already categorized (hash-based)",
)
@click.option(
    "--progress", is_flag=True, default=False, help="Show progress for long runs"
)
def categorize(input, output, categories, model, dedupe, progress):
    class Args:
        def __init__(self):
            self.input = input
            self.output = output
            self.categories = categories
            self.model = model
            self.dedupe = dedupe
            self.progress = progress

    args = Args()
    result = run_recording_categorize(args)
    raise SystemExit(result)


cli.add_command(transcribe)
cli.add_command(categorize)


def run_recording_transcribe(args) -> int:
    input_path = Path(args.input).expanduser()
    output_root = _resolve_output_root(input_path, args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.check_ffmpeg:
        _ensure_ffmpeg_available()

    if input_path.is_dir():
        input_files = [path for path in input_path.rglob("*") if path.is_file()]
        if not input_files:
            raise FileNotFoundError(f"No files found in directory: {input_path}")
        results = []
        for path in sorted(input_files):
            output_dir = output_root / _get_transcription_name(path)
            output_dir.mkdir(parents=True, exist_ok=True)
            results.append(_transcribe_single_file(path, output_dir, args))
        return 0 if all(code == 0 for code in results) else 1

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.is_file():
        output_dir = output_root / _get_transcription_name(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return _transcribe_single_file(input_path, output_dir, args)

    raise ValueError(f"Unsupported input path: {input_path}")


def _resolve_output_root(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser()
    if input_path.is_dir():
        return input_path
    return input_path.parent


def _get_transcription_name(path: Path) -> str:
    stat = path.stat()
    creation_time = stat.st_mtime
    dt = datetime.datetime.fromtimestamp(creation_time)
    return f"transcription-{dt.strftime('%Y-%m-%d-%H%M%S')}"


def _transcribe_single_file(
    source_path: Path,
    output_dir: Path,
    args,
) -> int:
    outputs = _build_output_paths(output_dir)
    if not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"Outputs already exist {existing}. Use --overwrite to replace them."
            )

    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faster-whisper is required. Install it to run transcription."
        ) from exc

    temp_audio: Path | None = None
    source_for_transcribe = source_path
    try:
        if _is_video_file(source_path):
            _ensure_ffmpeg_available()
            temp_audio = _extract_audio(source_path)
            source_for_transcribe = temp_audio

        click.echo(f"Processing: {source_path}")
        model = WhisperModel(args.whisper_model, device=args.device)
        segments_iter, info = model.transcribe(
            str(source_for_transcribe),
            language=args.language,
        )

        created_at = _utc_now_iso()
        source_hash = _sha256_file(source_path)
        segment_records = []

        start_time = time.monotonic()
        total_duration = getattr(info, "duration", None) or 0.0
        for idx, segment in enumerate(segments_iter, start=1):
            record = {
                "id": f"seg-{idx:04d}",
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "source": str(source_path),
                "model": args.whisper_model,
                "lang": info.language if info else args.language,
            }
            segment_records.append(record)
            _emit_progress(start_time, total_duration, segment.end)

        if segment_records:
            _emit_progress(
                start_time, total_duration, segment_records[-1]["end"], done=True
            )

        _write_segments(outputs["segments"], segment_records)
        chunk_records = _chunk_segments(
            segment_records,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
        )
        _write_chunks(outputs["chunks"], chunk_records)

        manifest = {
            "source_path": str(source_path),
            "source_hash": f"sha256:{source_hash}",
            "created_at": created_at,
            "model": args.whisper_model,
            "language": info.language if info else args.language,
            "device": args.device,
            "chunk_tokens": args.chunk_tokens,
            "chunk_overlap": args.chunk_overlap,
            "outputs": {
                "segments": outputs["segments"].name,
                "chunks": outputs["chunks"].name,
                "transcript": outputs["transcript"].name,
                "srt": outputs["srt"].name,
            },
        }
        outputs["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if args.emit in {"txt", "all"}:
            _write_transcript(outputs["transcript"], segment_records)
        if args.emit in {"srt", "all"}:
            _write_srt(outputs["srt"], segment_records)

        return 0
    finally:
        if temp_audio and temp_audio.exists():
            temp_audio.unlink()


def _build_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "segments": output_dir / "segments.jsonl",
        "chunks": output_dir / "chunks.jsonl",
        "manifest": output_dir / "manifest.json",
        "transcript": output_dir / "transcript.txt",
        "srt": output_dir / "transcript.srt",
    }


def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "ffmpeg not found in PATH. Install it (apt-get install ffmpeg or brew install ffmpeg)."
    )


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".flv",
        ".m4v",
        ".mpg",
        ".mpeg",
    }


def _extract_audio(source_path: Path) -> Path:
    temp_handle = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    temp_handle.close()
    temp_path = Path(temp_handle.name)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(temp_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        temp_path.unlink(missing_ok=True)
        message = result.stderr.strip() or "ffmpeg audio extraction failed"
        raise RuntimeError(message)
    return temp_path


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _chunk_segments(
    segments: list[dict],
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[dict]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive")

    chunks = []
    current_segments = []
    current_tokens = 0
    chunk_index = 1

    for segment in segments:
        tokens = _estimate_tokens(segment["text"])
        if current_segments and current_tokens + tokens > chunk_tokens:
            chunks.append(_build_chunk(chunk_index, current_segments, current_tokens))
            chunk_index += 1
            current_segments, current_tokens = _apply_overlap(
                current_segments, chunk_overlap
            )
        current_segments.append(segment)
        current_tokens += tokens

    if current_segments:
        chunks.append(_build_chunk(chunk_index, current_segments, current_tokens))

    return chunks


def _apply_overlap(
    segments: list[dict],
    overlap_tokens: int,
) -> tuple[list[dict], int]:
    if overlap_tokens <= 0:
        return [], 0
    kept = []
    total = 0
    for segment in reversed(segments):
        tokens = _estimate_tokens(segment["text"])
        if total + tokens > overlap_tokens and kept:
            break
        kept.append(segment)
        total += tokens
    kept.reverse()
    return kept, total


def _build_chunk(index: int, segments: list[dict], token_count: int) -> dict:
    return {
        "id": f"chunk-{index:04d}",
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "token_count": token_count,
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "segment_ids": [segment["id"] for segment in segments],
    }


def _write_segments(path: Path, segments: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps(segment, ensure_ascii=True) + "\n")


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=True) + "\n")


def _write_transcript(path: Path, segments: list[dict]) -> None:
    text = " ".join(segment["text"] for segment in segments).strip()
    path.write_text(text + "\n" if text else "", encoding="utf-8")


def _format_srt_timestamp(seconds: float) -> str:
    total_millis = int(seconds * 1000)
    hours, remainder = divmod(total_millis, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(path: Path, segments: list[dict]) -> None:
    lines = []
    for index, segment in enumerate(segments, start=1):
        start = _format_srt_timestamp(segment["start"])
        end = _format_srt_timestamp(segment["end"])
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(segment["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_daily_summary_path(input_dir: Path) -> Path:
    today = datetime.date.today().isoformat()
    return input_dir / f"daily-summary-{today}.md"


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _load_categories(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--categories must be a JSON array of strings") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("--categories must be a JSON array of strings")
    return value


def _dedupe_path(input_dir: Path) -> Path:
    return input_dir / ".categorize-dedupe.json"


def _load_dedupe_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return set()
    return {str(item) for item in data}


def _write_dedupe_hashes(path: Path, hashes: set[str]) -> None:
    path.write_text(json.dumps(sorted(hashes)), encoding="utf-8")


def _chunk_hash(chunk: dict) -> str:
    payload = f"{chunk.get('id', '')}|{chunk.get('text', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_category(response_text: str, categories: list[str]) -> str:
    normalized = response_text.strip()
    for category in categories:
        if normalized.lower() == category.lower():
            return category
    lowered = normalized.lower()
    for category in categories:
        if category.lower() in lowered:
            return category
    return categories[0]


def _format_hhmmss(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00:00"
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _append_to_markdown_section(path: Path, category: str, bullet: str) -> None:
    heading = f"## {category}"
    if path.exists():
        content = path.read_text(encoding="utf-8")
    else:
        title = f"# Daily Summary {datetime.date.today().isoformat()}"
        content = f"{title}\n"

    lines = content.splitlines()
    if heading not in lines:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(heading)
        lines.append(bullet)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    heading_index = lines.index(heading)
    insert_at = len(lines)
    for idx in range(heading_index + 1, len(lines)):
        if lines[idx].startswith("## "):
            insert_at = idx
            break
    lines.insert(insert_at, bullet)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_recording_categorize(args) -> int:
    input_path = Path(args.input).expanduser()
    if input_path.is_file():
        chunks_path = input_path
        input_dir = input_path.parent
    else:
        input_dir = input_path
        chunks_path = input_dir / "chunks.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

    categories = _load_categories(args.categories)
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_daily_summary_path(input_dir)
    )

    try:
        import llm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "llm is required for categorization. Install it to run recording-categorize."
        ) from exc

    model = llm.get_model(args.model)
    chunks = _load_chunks(chunks_path)

    dedupe_hashes: set[str] = set()
    if args.dedupe:
        dedupe_hashes = _load_dedupe_hashes(_dedupe_path(input_dir))

    total = len(chunks)
    processed = 0
    for index, chunk in enumerate(chunks, start=1):
        chunk_hash = _chunk_hash(chunk)
        if args.dedupe and chunk_hash in dedupe_hashes:
            continue

        prompt = (
            "Pick exactly one category from the list and respond with only the label.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Chunk:\n{chunk.get('text', '').strip()}"
        )
        response = model.prompt(prompt)
        response_text = (
            response.text()
            if hasattr(response, "text") and callable(response.text)
            else getattr(response, "text", str(response))
        )
        category = _extract_category(str(response_text), categories)

        start = _format_hhmmss(chunk.get("start"))
        end = _format_hhmmss(chunk.get("end"))
        chunk_id = chunk.get("id", "chunk")
        text = chunk.get("text", "").strip()
        bullet = f"- [{chunk_id} {start}-{end}] {text}"
        _append_to_markdown_section(output_path, category, bullet)

        if args.dedupe:
            dedupe_hashes.add(chunk_hash)

        processed += 1
        if args.progress:
            print(f"Processed {processed}/{total}", end="\r", file=sys.stderr)

    if args.progress and total > 0:
        print(" " * 30, end="\r", file=sys.stderr)
        print(f"Processed {processed}/{total}", file=sys.stderr)

    if args.dedupe:
        _write_dedupe_hashes(_dedupe_path(input_dir), dedupe_hashes)

    return 0


def _emit_progress(
    start_time: float,
    total_duration: float,
    current_end: float,
    done: bool = False,
) -> None:
    if total_duration <= 0:
        return
    elapsed = time.monotonic() - start_time
    progress = min(current_end / total_duration, 1.0)
    eta = (elapsed / progress - elapsed) if progress > 0 else 0
    percent = int(progress * 100)
    message = f"{percent:3d}% ETA {eta:6.1f}s"
    end_char = "\n" if done else "\r"
    print(message, end=end_char, file=sys.stderr, flush=True)


if __name__ == "__main__":
    cli()
