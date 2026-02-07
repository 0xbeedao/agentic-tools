#!/usr/bin/env python3
"""Generate INDEX.json files from markdown frontmatter."""

import json
from pathlib import Path

import click
import frontmatter


def get_file_metadata(file_path: Path, root_dir: Path) -> dict:
    """Extract all frontmatter metadata from a markdown file."""
    try:
        post = frontmatter.load(file_path)
        # Convert metadata to JSON-serializable format
        metadata = {}
        for key, value in post.metadata.items():
            # Handle non-serializable types (like dates)
            if hasattr(value, "isoformat"):
                metadata[key] = value.isoformat()
            else:
                metadata[key] = value

        # Add file path relative to root
        rel_path = str(file_path.relative_to(root_dir))
        return {"path": rel_path, "metadata": metadata}
    except Exception:
        # Return empty metadata for files that can't be parsed
        rel_path = str(file_path.relative_to(root_dir))
        return {"path": rel_path, "metadata": {}}


def generate_index(directory: Path, root_dir: Path, overwrite: bool = True) -> None:
    """Generate INDEX.json for a single directory level."""
    index_path = directory / "INDEX.json"

    if index_path.exists() and not overwrite:
        click.echo(
            f"Skipping {index_path} (already exists, use --overwrite to replace)"
        )
        return

    files_data = []

    # Get all markdown files in this directory only (not subdirectories)
    for md_file in sorted(directory.glob("*.md")):
        if md_file.name == "INDEX.json":
            continue

        file_data = get_file_metadata(md_file, root_dir)
        files_data.append(file_data)

    if files_data:
        index_content = json.dumps(files_data, indent=2, sort_keys=True)
        index_path.write_text(index_content + "\n", encoding="utf-8")
        click.echo(f"Generated {index_path} with {len(files_data)} entries")
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
    help="Overwrite existing INDEX.json files (default: True)",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Process subdirectories recursively (default: True)",
)
def main(directory: Path, overwrite: bool, recursive: bool) -> None:
    """Generate INDEX.json files from markdown frontmatter.

    DIRECTORY is the root directory to process.
    Each INDEX.json contains a map of files and their YAML frontmatter metadata.
    """
    # Process the root directory
    generate_index(directory, directory, overwrite)

    # Process subdirectories if recursive
    if recursive:
        for subdir in sorted(directory.rglob("*")):
            if subdir.is_dir():
                generate_index(subdir, directory, overwrite)


if __name__ == "__main__":
    main()
