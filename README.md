# Verify masked text

`verify_no_words.py` checks a pandas table for dictionary words in English (`en`), Simplified Chinese (`sc`), and Traditional Chinese (`tc`). A clean table has none of those words. Pandas is the only third-party dependency. Matching uses the Python standard library.

## Run

```bash
python verify_no_words.py data.csv --en wordlists/en.txt --sc wordlists/sc.txt --tc wordlists/tc.txt
```

Word lists and the CSV are UTF-8. A leading BOM is accepted.

| Exit code | Meaning |
| --- | --- |
| 0 | No dictionary words found |
| 1 | At least one dictionary word found |
| 2 | Bad path, unreadable file, or invalid argument |

Stdout prints one hit per line, then a count:

```text
1	note	en	apple	hello apple
1	note	sc	中国	中国人
1	note	sc	国人	中国人
en=1 sc=2 tc=0
```

Columns are `row`, `column`, `lang`, `word`, `snippet`. `row` is the 1-based data row (the header is not row 1). `lang` is `en`, `sc`, or `tc`.

`--report hits.tsv` writes the same rows with `DataFrame.to_csv` (tab-separated, UTF-8, with a header).

## Options

| Option | Default | Role |
| --- | --- | --- |
| `--en`, `--sc`, `--tc` | required | Word-list paths |
| `--columns` | string columns | Comma-separated names, for example `note,comment` |
| `--workers` | CPU count | Process count. `1` scans in the current process |
| `--min-len` | `2` | Drop dictionary entries shorter than this |
| `--match-limit` | `20` | Stop listing further matches from one cell. The cell still fails |

On Windows, run the scanner as a script (`python verify_no_words.py ...`) when `--workers` is greater than 1. The entry point is guarded with `if __name__ == "__main__"`.

## Word lists

Bundled lists are in `wordlists/`:

| File | Language | Entries |
| --- | --- | --- |
| `wordlists/en.txt` | English | 644,145 |
| `wordlists/sc.txt` | Simplified Chinese | 121,292 |
| `wordlists/tc.txt` | Traditional Chinese | 122,668 |

English is SCOWL 2020.12.07, every list under `final/`, casefolded. Chinese headwords are from CC-CEDICT (CC BY-SA 4.0). Sources, copyright, and the changes made to those files are in `wordlists/SOURCES.txt`.

```bash
python verify_no_words.py data.csv --en wordlists/en.txt --sc wordlists/sc.txt --tc wordlists/tc.txt
```

One word per line. Blank lines and lines that start with `#` are ignored. Text is normalized to NFC. English entries are matched case-insensitively.

```text
# English
apple
don't

# Simplified Chinese, in sc.txt
中国
国人
```

`--min-len 2` skips one-character entries such as English `a` / `I` and Chinese `的`. Pass `--min-len 1` to keep them.

## What is scanned

`pd.read_csv(..., encoding="utf-8-sig")` loads the file. The default scan covers string-like columns (`object`, `string`, `str`). Numbers stay numbers and are not turned into text. Blank cells and `NA` / `None` are skipped, so the literal `nan` is not treated as a word.

`--columns` overrides that filter. Only real strings inside those columns are checked.

Duplicate column names are rejected.

## How a word matches

English uses whole tokens. The pattern is letters, optionally with one apostrophe group such as `don't`. `cat` does not match inside `category`.

Chinese words are not split on spaces. Each language has its own character trie. The scanner walks every character offset and reports every full dictionary word, including overlaps: `中国人` can report both `中国` and `国人`. A word listed in both Chinese files is reported once per language.

Each cell stops after `--match-limit` hits. The snippet is a short window around the match, with whitespace collapsed onto one line.

Rows are split into chunks of 2000 and scanned with `ProcessPoolExecutor`. About `workers * 2` chunks are in flight. Each worker loads the three lists once. Pass `--workers 1` for a single-process scan.

## Call it from Python

```python
import pandas as pd
from verify_no_words import verify_dataframe

df = pd.read_csv("data.csv", encoding="utf-8-sig")
hits = verify_dataframe(
    df,
    "wordlists/en.txt",
    "wordlists/sc.txt",
    "wordlists/tc.txt",
    columns=["note"],
    workers=1,
)
```

`hits` is a list of dicts with keys `row`, `column`, `lang`, `word`, and `snippet`, sorted by those fields. An empty list means the scanned cells are clean.
