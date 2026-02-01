process-recordings ARGS:
    uv run automations/recording-processor/recording-processor.py {{ ARGS }}

md-frontmatter ARGS:
    uv run automations/markdown-frontmatter/markdown-frontmatter.py {{ ARGS }}
