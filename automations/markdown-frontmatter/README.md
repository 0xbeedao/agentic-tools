# Markdown Frontmatter

A simple python script accepting a directory argument.  it then walks the directory and non-hidden children, operating on all markdown files.

If the file already has all three required fields, and all have data, leave it alone.
Else, ensure all are there and have data by using the "llm" module and calling the default "gpt4.2" model (override with a switch) with the file contents to fill in the "tags" and "description".  Fill in the date by the file modified time if not already given

# Fields:

these should be preceded and followed by a line with three dashes, making a frontmatter.

- date: yyyy-mm-dd HH:MM
- tags: [comma,sep,tags]
- description: a one sentence description of the contents
- author: author name (optional)
- url: url of original (optional)


## Optional fields

Add if processing by llm, and if obvious - but its lack does not cause the file to be read by the llm

## Usage

Run from the repo root:

```bash
python automations/markdown-frontmatter/markdown-frontmatter.py /path/to/notes --dry-run
python automations/markdown-frontmatter/markdown-frontmatter.py /path/to/notes --model gpt4.2
```
