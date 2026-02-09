#!/usr/bin/env nu

# Check if there are any files in the recordings directory
let recordings_dir = ("~/sync/recordings" | path expand)
let files = (ls $recordings_dir | where type == file and name =~ ".mp3$")

if ($files | is-empty) {
    echo "No mp3 files found in ~/sync/recordings - nothing to process"
    exit 0
}

# Run transcribe command
echo "Running transcription..."
let transcribe_result = (do {
    ^recording-processor --archive ("~/Bak/recordings" | path expand) transcribe --input ("~/sync/recordings" | path expand) --output ("~/tmp/transcripts" | path expand)
} | complete)

if $transcribe_result.exit_code != 0 {
    echo "Transcription failed with exit code: ($transcribe_result.exit_code)"
    echo $transcribe_result.stderr
    exit $transcribe_result.exit_code
}

echo $transcribe_result.stdout

# Run categorize command
echo "Running categorization..."
let categorize_result = (do {
    ^recording-processor --archive ("~/notes/3_Resources/transcripts" | path expand) categorize --input ("~/tmp/transcripts" | path expand) --outputdir ("~/notes/4_Archives/Journal/" | path expand)
} | complete)

if $categorize_result.exit_code != 0 {
    echo "Categorization failed with exit code: ($categorize_result.exit_code)"
    echo $categorize_result.stderr
    exit $categorize_result.exit_code
}

echo $categorize_result.stdout

# Success
let now = (date now | format date "%Y-%m-%d %H:%M:%S")
echo "[($now)] done success"
exit 0
