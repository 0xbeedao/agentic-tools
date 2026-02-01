import datetime
import json
import os
import re
from pathlib import Path

import click


_MANAGED_KEYS = ("date", "tags", "description", "author", "url")
_RE_KEY_VALUE = re.compile(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$")


def _is_hidden_path(path: Path) -> bool:
    for part in path.parts:
        if part in {".", ".."}:
            continue
        if part.startswith("."):
            return True
    return False


def _iter_markdown_files(root: Path) -> list[Path]:
    results: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        current = Path(current_root)
        if _is_hidden_path(current.relative_to(root)):
            continue
        for name in filenames:
            if name.startswith("."):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".md", ".markdown"}:
                continue
            results.append(current / name)
    return sorted(results)


def _split_frontmatter(text: str) -> tuple[list[str] | None, str]:
    lines = text.splitlines()
    if not lines:
        return None, ""
    if lines[0].strip() != "---":
        return None, text
    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        return None, text
    fm_lines = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :])
    if text.endswith("\n"):
        body += "\n"
    return fm_lines, body


def _extract_managed_values(fm_lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in fm_lines:
        match = _RE_KEY_VALUE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        if key not in _MANAGED_KEYS:
            continue
        if key in values:
            continue
        values[key] = match.group(2).strip()
    return values


def _remove_managed_blocks(fm_lines: list[str]) -> list[str]:
    """Remove managed keys and their indented block values from frontmatter."""

    remaining: list[str] = []
    skipping_block = False
    for line in fm_lines:
        if skipping_block:
            if not line.strip():
                continue
            if line.startswith(" ") or line.startswith("\t"):
                continue
            skipping_block = False

        match = _RE_KEY_VALUE.match(line)
        if match and match.group(1).strip() in _MANAGED_KEYS:
            raw_value = match.group(2).strip()
            if raw_value in {"", "|", ">", "|-", ">-"}:
                skipping_block = True
            continue

        remaining.append(line)

    return remaining


def _format_date_from_mtime(path: Path) -> str:
    dt = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M")


def _strip_wrapping_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _parse_tags(value: str) -> list[str]:
    v = _strip_wrapping_quotes(value).strip()
    if not v:
        return []
    if v == "[]":
        return []
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1]
        parts = [p.strip() for p in inner.split(",")]
    elif "," in v:
        parts = [p.strip() for p in v.split(",")]
    else:
        parts = [v]
    tags = []
    for part in parts:
        cleaned = _strip_wrapping_quotes(part).strip()
        if cleaned:
            tags.append(cleaned)
    return tags


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        t = tag.strip().lower()
        t = t.lstrip("#")
        t = re.sub(r"\s+", "-", t)
        t = re.sub(r"[^a-z0-9\-_/]", "", t)
        t = t.strip("-_")
        if not t or t in seen:
            continue
        seen.add(t)
        normalized.append(t)
    return normalized


def _yaml_quote_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_tags_inline(tags: list[str]) -> str:
    return "[" + ", ".join(tags) + "]"


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    raise ValueError("Model did not return a JSON object")


def _llm_enrich(*, model_name: str, markdown: str, max_chars: int) -> dict:
    try:
        import llm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "llm is required. Install it to run this tool."
        ) from exc

    try:
        model = llm.get_model(model_name)
    except Exception as exc:
        available = []
        try:
            available = [m.model_id for m in llm.get_models()]
        except Exception:
            available = []
        hint = ""
        if available:
            hint = "\nAvailable models: " + ", ".join(sorted(set(available))[:25])
        raise RuntimeError(f"Unknown/unavailable model: {model_name}{hint}") from exc

    excerpt = markdown
    if max_chars > 0 and len(excerpt) > max_chars:
        half = max_chars // 2
        excerpt = excerpt[:half] + "\n\n[...TRUNCATED...]\n\n" + excerpt[-half:]

    system = (
        "You generate frontmatter metadata for markdown files. Return strict JSON only."
    )
    prompt = (
        "Given this markdown document, fill in missing metadata.\n\n"
        "Return a single JSON object with keys:\n"
        "- tags: array of 3-8 short, lowercase tags (no spaces; use hyphens)\n"
        "- description: one sentence describing the document\n"
        "- author: author name if obvious, else empty string\n"
        "- url: source URL if obvious, else empty string\n\n"
        "Do not wrap JSON in markdown fences.\n\n"
        f"MARKDOWN:\n{excerpt}"
    )

    response = model.prompt(prompt, system=system, stream=False)
    response_text = (
        response.text()
        if hasattr(response, "text") and callable(response.text)
        else getattr(response, "text", str(response))
    )
    return _extract_json_object(str(response_text))


def _ensure_frontmatter(
    *,
    path: Path,
    text: str,
    model_name: str,
    max_chars: int,
) -> tuple[str, bool, bool]:
    fm_lines, body = _split_frontmatter(text)

    values: dict[str, str] = {}
    if fm_lines is not None:
        values = _extract_managed_values(fm_lines)

    date_value = _strip_wrapping_quotes(values.get("date", "")).strip()
    tags_value = values.get("tags", "").strip()
    description_value = _strip_wrapping_quotes(values.get("description", "")).strip()
    author_value = _strip_wrapping_quotes(values.get("author", "")).strip()
    url_value = _strip_wrapping_quotes(values.get("url", "")).strip()

    tags_list = _normalize_tags(_parse_tags(tags_value))
    needs_llm = not tags_list or not description_value

    llm_used = False
    if needs_llm:
        payload = _llm_enrich(
            model_name=model_name, markdown=body or text, max_chars=max_chars
        )
        llm_used = True
        if not tags_list:
            raw_tags = payload.get("tags", [])
            if isinstance(raw_tags, list):
                tags_list = _normalize_tags([str(t) for t in raw_tags])
        if not description_value:
            raw_desc = payload.get("description", "")
            if isinstance(raw_desc, str):
                description_value = raw_desc.strip()
        if not author_value:
            raw_author = payload.get("author", "")
            if isinstance(raw_author, str):
                author_value = raw_author.strip()
        if not url_value:
            raw_url = payload.get("url", "")
            if isinstance(raw_url, str):
                url_value = raw_url.strip()

    if not date_value:
        date_value = _format_date_from_mtime(path)

    if not tags_list:
        raise ValueError("Could not determine non-empty tags")
    if not description_value:
        raise ValueError("Could not determine non-empty description")

    managed_lines: list[str] = []
    managed_lines.append(f"date: {_yaml_quote_string(date_value)}")
    managed_lines.append(f"tags: {_format_tags_inline(tags_list)}")
    managed_lines.append(f"description: {_yaml_quote_string(description_value)}")
    if author_value:
        managed_lines.append(f"author: {_yaml_quote_string(author_value)}")
    if url_value:
        managed_lines.append(f"url: {_yaml_quote_string(url_value)}")

    if fm_lines is not None:
        remaining = _remove_managed_blocks(fm_lines)
        fm_out_lines = managed_lines + remaining
    else:
        fm_out_lines = managed_lines

    new_text = "---\n" + "\n".join(fm_out_lines).rstrip() + "\n---\n\n" + (body or text)
    if not new_text.endswith("\n"):
        new_text += "\n"

    changed = new_text != text
    return new_text, changed, llm_used


@click.command()
@click.argument(
    "directory", type=click.Path(path_type=Path, exists=True, file_okay=False)
)
@click.option(
    "--model",
    default="openrouter/minimax/minimax-m2.1",
    show_default=True,
    help="llm model name/alias to use",
)
@click.option(
    "--max-chars",
    default=20000,
    show_default=True,
    type=int,
    help="Max markdown characters sent to the model (0 = unlimited)",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Show changes without writing"
)
@click.option(
    "--verbose", is_flag=True, default=False, help="Print each processed file"
)
def cli(
    directory: Path, model: str, max_chars: int, dry_run: bool, verbose: bool
) -> None:
    root = directory.expanduser().resolve()
    paths = _iter_markdown_files(root)

    changed_count = 0
    llm_count = 0
    skipped_count = 0
    error_count = 0

    for path in paths:
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            error_count += 1
            click.echo(f"[error] {path}: not utf-8", err=True)
            continue

        try:
            updated, changed, llm_used = _ensure_frontmatter(
                path=path,
                text=original,
                model_name=model,
                max_chars=max_chars,
            )
        except Exception as exc:
            error_count += 1
            click.echo(f"[error] {path}: {exc}", err=True)
            continue

        if llm_used:
            llm_count += 1

        if not changed:
            skipped_count += 1
            if verbose:
                click.echo(f"[ok] {path}")
            continue

        changed_count += 1
        if verbose or dry_run:
            click.echo(f"[update] {path}")

        if not dry_run:
            path.write_text(updated, encoding="utf-8")

    click.echo(
        f"Processed {len(paths)} markdown files: {changed_count} updated, {skipped_count} unchanged, {error_count} errors. LLM calls: {llm_count}"
    )
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
