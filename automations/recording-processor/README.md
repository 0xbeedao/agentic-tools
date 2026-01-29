# recording processor

An automation that:

- processes all mp3 files in WATCHED DIR
- moves these files to TARGET DIR, renaming to "recording-(short orig create date with time).mp3"
- sends the mp3s to "faster-whisper" (already globally installed)
- saves the output to the same filename base as markdown
