import argparse
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# Exact separator: two newlines, [PAGE_NUMBER], 48 hyphens, two newlines.
SEPARATOR = re.compile(r"\n\n\[([0-9]+)\]" + re.escape("-" * 48) + r"\n\n")
CANDIDATE = re.compile(r"\n\n\[[^\]\n]*\]-+\n\n")


def prepare_book(source: Path, destination: Path, title: str, authors: str) -> None:
    text = source.read_text(encoding="utf-8")
    matches = list(SEPARATOR.finditer(text))
    if not matches:
        raise ValueError(f"{source}: no valid page separators found")

    candidates = list(CANDIDATE.finditer(text))
    if len(candidates) != len(matches):
        raise ValueError(f"{source}: malformed page separator found")

    numbers = [int(match.group(1)) for match in matches]
    if any(b <= a for a, b in zip(numbers, numbers[1:])):
        raise ValueError(f"{source}: page numbers must increase strictly")

    # The text before the first separator has no validated page number.
    # Keep it explicitly unnumbered rather than assigning a guessed page.
    sections = [
        f"# {title}\n\nAuthors: {authors}\n\n"
        "Page labels below come from validated source separators.\n"
    ]
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(f"\n## Unnumbered front matter\n\n{prefix}\n")

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            raise ValueError(f"{source}: page {match.group(1)} is empty")
        sections.append(
            f"\n## Source page {match.group(1)}\n\n"
            f"Book: {title}\nAuthors: {authors}\n"
            f"Source page: {match.group(1)}\n\n{body}\n"
        )

    destination.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path, help="Authorized Markdown book")
    parser.add_argument("--title", required=True)
    parser.add_argument("--authors", required=True)
    parser.add_argument(
        "--vector-store-id",
        help="Existing store ID; omit to create a new store",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the operator's environment")

    source = args.markdown.resolve()
    if not source.is_file():
        parser.error(f"File not found: {source}")

    prepared = source.with_name(source.stem + ".page-labelled.md")
    prepare_book(source, prepared, args.title, args.authors)

    client = OpenAI()
    store_id = args.vector_store_id
    if not store_id:
        store_id = client.vector_stores.create(name="book_research").id

    with prepared.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="assistants")

    client.vector_stores.files.create(
        vector_store_id=store_id,
        file_id=uploaded.id,
    )

    while True:
        indexed = client.vector_stores.files.retrieve(
            vector_store_id=store_id,
            file_id=uploaded.id,
        )
        if indexed.status == "completed":
            break
        if indexed.status in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Indexing did not complete; status: {indexed.status}"
            )
        time.sleep(3)

    print(f"Indexed: {prepared}")
    print(f"OpenAI file ID: {uploaded.id}")
    print(f"OPENAI_VECTOR_STORE_ID = {store_id}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
