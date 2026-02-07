process-recordings ARGS:
    uv run automations/recording_processor/recording_processor.py {{ ARGS }}

md-frontmatter ARGS:
    uv run automations/markdown-frontmatter/markdown-frontmatter.py {{ ARGS }}
