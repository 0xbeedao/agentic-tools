#!/usr/bin/env python3
"""Generate TOC.md files from markdown frontmatter descriptions."""

import re
from pathlib import Path

import click
import yaml


def extract_frontmatter(content: str) -> dict | None:
    """Extract YAML frontmatter from markdown content."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    return None


def get_description(file_path: Path) -> str | None:
    """Extract description from file's YAML frontmatter."""
    try:
        content = file_path.read_text(encoding="utf-8")
        frontmatter = extract_frontmatter(content)
        if frontmatter and "description" in frontmatter:
            return frontmatter["description"]
    except (IOError, UnicodeDecodeError):
        pass
    return None


def generate_toc(directory: Path, overwrite: bool = True) -> None:
    """Generate TOC.md for a single directory level."""
    toc_path = directory / "TOC.md"

    if toc_path.exists() and not overwrite:
        click.echo(f"Skipping {toc_path} (already exists, use --overwrite to replace)")
        return

    entries = []

    # Get all markdown files in this directory only (not subdirectories)
    for md_file in sorted(directory.glob("*.md")):
        if md_file.name == "TOC.md":
            continue

        description = get_description(md_file)
        if description:
            # Create relative link
            rel_link = md_file.name
            entries.append(f"- [{description}]({rel_link})")

    if entries:
        toc_content = "\n".join(entries) + "\n"
        toc_path.write_text(toc_content, encoding="utf-8")
        click.echo(f"Generated {toc_path} with {len(entries)} entries")
    else:
        click.echo(f"No entries found for {directory}")


@click.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    help="Overwrite existing TOC.md files (default: True)",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Process subdirectories recursively (default: True)",
)
def main(directory: Path, overwrite: bool, recursive: bool) -> None:
    """Generate TOC.md files from markdown frontmatter descriptions.

    DIRECTORY is the root directory to process.
    """
    # Process the root directory
    generate_toc(directory, overwrite)

    # Process subdirectories if recursive
    if recursive:
        for subdir in sorted(directory.rglob("*")):
            if subdir.is_dir():
                generate_toc(subdir, overwrite)


if __name__ == "__main__":
    main()
