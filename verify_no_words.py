"""Verify that a pandas DataFrame contains no dictionary words.

Word lists are local text files. English matches whole tokens. Simplified
and Traditional Chinese match through character tries, because those words
are not split by spaces. Pandas is the only third-party dependency.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import pandas as pd

_EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_END = object()
_CHUNK_ROWS = 2000
_HIT_COLUMNS = ("row", "column", "lang", "word", "snippet")
_STATE: dict = {}


def _load_words(path: str, min_len: int, *, casefold: bool) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            word = unicodedata.normalize("NFC", word)
            if casefold:
                word = word.casefold()
            if len(word) < min_len or word in seen:
                continue
            seen.add(word)
            words.append(word)
    return words


def _build_trie(words: list[str]) -> dict:
    root: dict = {}
    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node[_END] = word
    return root


def _init_worker(en_path: str, sc_path: str, tc_path: str, min_len: int) -> None:
    _STATE["min_len"] = min_len
    _STATE["en"] = set(_load_words(en_path, min_len, casefold=True))
    _STATE["sc"] = _build_trie(_load_words(sc_path, min_len, casefold=False))
    _STATE["tc"] = _build_trie(_load_words(tc_path, min_len, casefold=False))


def _readable_utf8(path: str) -> str:
    full_path = os.path.abspath(path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    with open(full_path, encoding="utf-8-sig") as handle:
        for _line in handle:
            pass
    return full_path


def _snippet(text: str, start: int, length: int, radius: int = 12) -> str:
    left = max(0, start - radius)
    right = min(len(text), start + length + radius)
    return " ".join(text[left:right].split())


def _collect_zh(text: str, trie: dict, lang: str, hits: list, limit: int) -> None:
    size = len(text)
    for start in range(size):
        node = trie
        for index in range(start, size):
            node = node.get(text[index])
            if node is None:
                break
            word = node.get(_END)
            if word is None:
                continue
            hits.append((lang, word, _snippet(text, start, len(word))))
            if len(hits) >= limit:
                return


def _scan_cell(text: str, match_limit: int) -> list[tuple[str, str, str]]:
    text = unicodedata.normalize("NFC", text)
    hits: list[tuple[str, str, str]] = []
    min_len = _STATE["min_len"]
    english = _STATE["en"]
    for match in _EN_WORD.finditer(text):
        token = match.group(0)
        if len(token) < min_len or token.casefold() not in english:
            continue
        hits.append(("en", token, _snippet(text, match.start(), len(token))))
        if len(hits) >= match_limit:
            return hits
    for lang in ("sc", "tc"):
        _collect_zh(text, _STATE[lang], lang, hits, match_limit)
        if len(hits) >= match_limit:
            return hits
    return hits


def _cell_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _is_text_dtype(dtype: object) -> bool:
    if isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    name = getattr(dtype, "name", "")
    if name in {"object", "string", "str"}:
        return True
    return bool(pd.api.types.is_string_dtype(dtype))


def _selected_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if df.columns.duplicated().any():
        raise ValueError("DataFrame has duplicate column names")
    if columns is None:
        return [column for column in df.columns if _is_text_dtype(df[column].dtype)]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(map(str, missing)))
    chosen: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            seen.add(column)
            chosen.append(column)
    return chosen


def _iter_chunks(df: pd.DataFrame, columns: list[str], chunk_rows: int):
    arrays = {
        column: df[column].to_numpy(dtype=object, na_value=None) for column in columns
    }
    total = len(df)
    for start in range(0, total, chunk_rows):
        rows = []
        end = min(start + chunk_rows, total)
        for pos in range(start, end):
            cells = []
            for column in columns:
                text = _cell_text(arrays[column][pos])
                if text is not None:
                    cells.append((column, text))
            if cells:
                rows.append((pos + 1, cells))
        if rows:
            yield rows


def _scan_chunk(rows: list, match_limit: int) -> list[dict]:
    found = []
    for row_number, cells in rows:
        for column, text in cells:
            for lang, word, snippet in _scan_cell(text, match_limit):
                found.append(
                    {
                        "row": row_number,
                        "column": column,
                        "lang": lang,
                        "word": word,
                        "snippet": snippet,
                    }
                )
    return found


def _scan_parallel(chunk_iter, workers: int, initargs: tuple, match_limit: int) -> list[dict]:
    try:
        first = next(chunk_iter)
    except StopIteration:
        return []
    hits: list[dict] = []
    max_pending = workers * 2
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=initargs,
    ) as executor:
        pending = {executor.submit(_scan_chunk, first, match_limit)}
        while len(pending) < max_pending:
            try:
                pending.add(executor.submit(_scan_chunk, next(chunk_iter), match_limit))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                hits.extend(future.result())
            while len(pending) < max_pending:
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(_scan_chunk, chunk, match_limit))
    return hits


def _sort_hits(hits: list[dict]) -> list[dict]:
    return sorted(
        hits,
        key=lambda hit: (
            hit["row"],
            str(hit["column"]),
            hit["lang"],
            hit["word"],
            hit["snippet"],
        ),
    )


def verify_dataframe(
    df: pd.DataFrame,
    en_path: str,
    sc_path: str,
    tc_path: str,
    columns: list[str] | None = None,
    workers: int | None = None,
    min_len: int = 2,
    match_limit: int = 20,
) -> list[dict]:
    """Return dictionary hits in string cells.

    Each hit has row, column, lang, word, and snippet. row is the 1-based
    position in df. lang is en, sc, or tc. Any hit means the table failed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if workers is None:
        workers = os.cpu_count() or 1
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    if match_limit < 1:
        raise ValueError("match_limit must be >= 1")

    paths = tuple(_readable_utf8(path) for path in (en_path, sc_path, tc_path))
    chosen = _selected_columns(df, columns)
    if not chosen:
        return []
    chunks = _iter_chunks(df, chosen, _CHUNK_ROWS)
    initargs = (paths[0], paths[1], paths[2], min_len)
    if workers == 1:
        _init_worker(*initargs)
        hits: list[dict] = []
        for chunk in chunks:
            hits.extend(_scan_chunk(chunk, match_limit))
        return _sort_hits(hits)
    return _sort_hits(_scan_parallel(chunks, workers, initargs, match_limit))


def _field(value: object) -> str:
    return " ".join(str(value).split())


def _format_hit(hit: dict) -> str:
    return "\t".join(_field(hit[key]) for key in _HIT_COLUMNS)


def _count_line(hits: list[dict]) -> str:
    counts = {"en": 0, "sc": 0, "tc": 0}
    for hit in hits:
        counts[hit["lang"]] += 1
    return f"en={counts['en']} sc={counts['sc']} tc={counts['tc']}"


def _hits_frame(hits: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(hits), columns=list(_HIT_COLUMNS))


def _parse_columns(value: str | None) -> list[str] | None:
    if value is None:
        return None
    columns = [part.strip() for part in value.split(",")]
    if not columns or any(not part for part in columns):
        raise ValueError("columns must be a comma-separated list of names")
    return columns


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check a CSV for dictionary words in English, Simplified Chinese, "
            "and Traditional Chinese."
        )
    )
    parser.add_argument("csv", help="CSV file to scan")
    parser.add_argument("--en", required=True, help="English word list, one word per line")
    parser.add_argument("--sc", required=True, help="Simplified Chinese word list")
    parser.add_argument("--tc", required=True, help="Traditional Chinese word list")
    parser.add_argument(
        "--columns",
        help="Comma-separated columns to scan. Default: string columns.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process count. Default: CPU count. Use 1 to scan in this process.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Ignore dictionary entries shorter than this length. Default: 2.",
    )
    parser.add_argument(
        "--match-limit",
        type=int,
        default=20,
        help="Maximum matches reported per cell. Default: 20.",
    )
    parser.add_argument("--report", help="Write the hit rows to this TSV path")
    args = parser.parse_args(argv)

    try:
        csv_path = _readable_utf8(args.csv)
        columns = _parse_columns(args.columns)
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        hits = verify_dataframe(
            frame,
            args.en,
            args.sc,
            args.tc,
            columns=columns,
            workers=args.workers,
            min_len=args.min_len,
            match_limit=args.match_limit,
        )
        _configure_stdout()
        for hit in hits:
            print(_format_hit(hit))
        print(_count_line(hits))
        if args.report:
            _hits_frame(hits).to_csv(args.report, sep="\t", index=False, encoding="utf-8")
    except (
        OSError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
