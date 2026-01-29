import json
from argparse import Namespace
from pathlib import Path

import pytest

import main


def _build_args(input_path: Path, output: Path | None = None) -> Namespace:
    return Namespace(
        input=str(input_path),
        output=str(output) if output else None,
        whisper_model="base",
        language=None,
        device="auto",
        emit="bundle",
        chunk_tokens=1000,
        chunk_overlap=100,
        overwrite=False,
        batch_mode="sequential",
        config=None,
        check_ffmpeg=False,
    )


def test_resolve_output_root_defaults_to_input_parent(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    assert main._resolve_output_root(input_path, None) == tmp_path


def test_resolve_output_root_uses_output_arg(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    output_root = tmp_path / "outputs"
    assert main._resolve_output_root(input_path, str(output_root)) == output_root


def test_transcribe_rejects_missing_input_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    args = _build_args(missing)
    with pytest.raises(FileNotFoundError, match="Input not found"):
        main.run_recording_transcribe(args)


def test_transcribe_rejects_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    args = _build_args(empty_dir)
    with pytest.raises(FileNotFoundError, match="No files found"):
        main.run_recording_transcribe(args)


def test_output_formatting_golden(tmp_path: Path) -> None:
    segments = [
        {
            "id": "seg-0001",
            "start": 0.0,
            "end": 1.2,
            "text": "Hello world",
            "source": "input.wav",
            "model": "base",
            "lang": "en",
        },
        {
            "id": "seg-0002",
            "start": 1.2,
            "end": 2.4,
            "text": "Second line",
            "source": "input.wav",
            "model": "base",
            "lang": "en",
        },
    ]

    chunks = main._chunk_segments(segments, chunk_tokens=10, chunk_overlap=0)
    expected_chunks = [
        {
            "id": "chunk-0001",
            "start": 0.0,
            "end": 2.4,
            "token_count": 4,
            "text": "Hello world Second line",
            "segment_ids": ["seg-0001", "seg-0002"],
        }
    ]
    assert chunks == expected_chunks

    segments_path = tmp_path / "segments.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    transcript_path = tmp_path / "transcript.txt"
    srt_path = tmp_path / "transcript.srt"

    main._write_segments(segments_path, segments)
    main._write_chunks(chunks_path, chunks)
    main._write_transcript(transcript_path, segments)
    main._write_srt(srt_path, segments)

    expected_segments = (
        "\n".join(json.dumps(segment, ensure_ascii=True) for segment in segments) + "\n"
    )
    expected_chunks_text = json.dumps(expected_chunks[0], ensure_ascii=True) + "\n"
    expected_transcript = "Hello world Second line\n"
    expected_srt = "\n".join(
        [
            "1",
            "00:00:00,000 --> 00:00:01,200",
            "Hello world",
            "",
            "2",
            "00:00:01,200 --> 00:00:02,400",
            "Second line",
            "",
        ]
    )

    assert segments_path.read_text(encoding="utf-8") == expected_segments
    assert chunks_path.read_text(encoding="utf-8") == expected_chunks_text
    assert transcript_path.read_text(encoding="utf-8") == expected_transcript
    assert srt_path.read_text(encoding="utf-8") == expected_srt
