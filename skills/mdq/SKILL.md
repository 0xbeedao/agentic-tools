---
name: mdq
description: Query and extract specific elements from Markdown documents using mdq (jq for Markdown). Use when parsing documentation, PR templates, Tickets, GitHub issues, or extracting structured data from Markdown files.
---

# mdq - Markdown Query Tool

## What This Skill Does

The `mdq` skill enables querying and extracting specific elements from Markdown documents. Like `jq` does for JSON, `mdq` provides powerful filtering capabilities for Markdown files including:

- Headings and sections
- Lists (ordered and unordered)
- Task lists (completed/uncompleted)
- Links and images
- Block quotes
- Code blocks
- Tables (with header/row filtering)
- Front matter (YAML/TOML)
- Plain paragraphs

## When to Use This Skill

Use `mdq` when you need to:

- Extract structured information from documentation
- Parse GitHub PR/issue templates
- Validate checklist completion in Markdown templates
- Find specific sections or elements in README files
- Query tables for specific data points
- Extract links or references from Markdown
- Parse front matter from Markdown files

## Capabilities

### Basic Selection

```bash
# Select sections containing specific text
mdq '# usage'

# Chain filters (e.g., sections containing "usage", then list items within)
mdq '# usage | -'

# Select uncompleted tasks
mdq '- [ ]'

# Select completed tasks
mdq '- [x]'

# Select any task
mdq '- [?]'
```

### Element Types

| Element | Syntax |
|---------|--------|
| Sections | `# title text` |
| Lists | `- unordered list item text` |
| Ordered lists | `1. ordered list item text` |
| Tasks | `- [ ]` or `- [x]` or `- [?]` |
| Links | `[display text](url)` |
| Images | `![alt text](url)` |
| Block quotes | `> block quote text` |
| Code blocks | `` ```language <code block text>`` `` |
| Tables | `:-: header text :-: row text` |
| Front matter | `+++[toml\|yaml] front matter text` |
| Paragraphs | `P: paragraph text` |

### Text Matching Patterns

- **Unquoted string**: `keyword` (case-insensitive, starts with letter)
- **Quoted string**: `"exact match"` (case-sensitive)
- **Anchored strings**: `^starts-with`, `ends-with$`
- **Regex**: `/pattern/` or `!s/regex/replacement/`
- **Any**: `*` or omit text

### Output Modes

```bash
# Default Markdown output
cat file.md | mdq '# introduction'

# JSON output (for integration with other tools)
cat file.md | mdq --output json '# ticket | [](^https://tickets.example.com/[A-Z]+-\d+$)'

# Quiet mode (exit code: 0 if match found, non-0 otherwise)
if echo "$ISSUE_TEXT" | mdq -q '- [x] I have searched for existing issues'; then
  echo "Found completed checklist"
fi
```

### Table Filtering

Tables can be filtered to select specific columns and rows:

```bash
# Select specific columns (On-Call and Alice columns from oncall schedule)
cat oncall.md | mdq ':-: /On-Call|Alice/:-: *'

# Select specific row by date
cat oncall.md | mdq ':-: * :-: 2024-01-15'
```

## Instructions

### Step 1: Identify the Query

When the user requests Markdown parsing:

1. Determine what elements they want to extract (headings, lists, tasks, tables, etc.)
2. Identify any text patterns to match (keywords, regex, exact strings)
3. Consider if filters need to be chained (e.g., section → list items)

### Step 2: Build the mdq Command

Construct the query using mdq syntax:

```bash
# Basic usage pattern
mdq '<query>'

# With input file
cat <file.md> | mdq '<query>'

# With JSON output for further processing
cat <file.md> | mdq --output json '<query>' | jq '.items[]...'
```

### Step 3: Execute and Present Results

Run the command and present results clearly:

```bash
# Example: Extract all completed tasks from a PR template
cat PR_TEMPLATE.md | mdq '- [x]'
```

For structured queries (especially with tables), interpret the output and provide context.

## Examples

### Example 1: Validate PR Checklist Completion

```bash
# Check if user confirmed they searched for existing issues
if echo "$PR_BODY" | mdq -q '- [x] I have searched for existing issues'; then
  echo "Checklist validated"
else
  echo "Missing checklist item"
fi
```

### Example 2: Extract Ticket Link from PR

```bash
# Extract ticket URL and use with jq
TICKET_URL="$(echo "$PR_TEXT" \
  | mdq --output json '# Ticket | [](^https://tickets.example.com/[A-Z]+-\d+$)' \
  | jq -r '.items[].link.url')"

echo "Ticket: $TICKET_URL"
```

### Example 3: Find All Uncompleted Tasks

```bash
# Find all unchecked items in a document
cat DOCUMENT.md | mdq '- [ ]'
```

### Example 4: Extract Specific Section

```bash
# Extract the full "API Reference" section
cat README.md | mdq '# API Reference'
```

### Example 5: Query On-Call Schedule Table

```bash
# Find who's on call for a specific week
cat oncall.md | mdq ':-: * :-: 2024-01-15'

# Find which weeks a specific person is on call
cat oncall.md | mdq ':-: /On-Call|Alice/:-: *'
```

### Example 6: Extract Links from Documentation

```bash
# Extract all GitHub links from README
cat README.md | mdq '[](^https://github.com/)'

# Extract all anchor links with specific text
cat README.md | mdq '[learn more]'
```

## Constraints and Limitations

- mdq operates on well-formed Markdown; malformed documents may produce unexpected results
- Regex matching follows Rust regex syntax (similar to PCRE but not identical)
- When using regex replacement (`!s/regex/replacement/`), consult the user manual for caveats
- For very large files, consider using specific filters to reduce output size
- Always verify exit codes when using `-q` mode for conditional logic

## Verification

After running mdq queries:

1. **Check exit code**: Ensure command succeeded (exit code 0)
2. **Validate output**: Results match expected format (Markdown or JSON)
3. **Test edge cases**: Verify behavior with empty results, multiple matches, etc.
4. **Confirm matches**: For critical queries, visually inspect a sample of results

## Common Patterns

### Finding Uncompleted Work

```bash
# All - [ ] tasks in a document
mdq '- [ ]'
```

### Extracting Links to External Resources

```bash
# All links to external domains (not relative)
mdq '[](^https?://)'
```

### Parsing Front Matter

```bash
# Extract YAML/TOML front matter
mdq '+++yaml'  # or '+++toml'
```

### Code Block Discovery

```bash
# Find Python code blocks
echo -e '```python\nprint("hello")\n```' | mdq '```python'
```

## References

- **Repository**: https://github.com/yshavit/mdq
- **Documentation**: https://github.com/yshavit/mdq/wiki/Full-User-Manual
- **Tutorial**: https://github.com/yshavit/mdq/wiki/Tutorial
- **Live Playground**: https://yshavit.github.io/mdq-playground
- **Installation**: `brew install mdq` (Mac/Linux) or download from releases
- **Requirements**: Rust ≥ 1.80.1 if building from source with cargo

## Troubleshooting

### Command Not Found
Install mdq via brew, download binary, or use Docker:
```bash
brew install mdq
# OR
docker pull yshavit/mdq
cat file.md | docker run --rm -i yshavit/mdq '# section'
```

### No Results
- Verify the query syntax matches the document structure
- Check text matching (case sensitivity, regex patterns)
- Test with `*` wildcard to see all elements of a type

### Unexpected Results
- Consult the full user manual for edge cases
- Use live playground to test queries interactively
- Check for malformed Markdown in the source file
