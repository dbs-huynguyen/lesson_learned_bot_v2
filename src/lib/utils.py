import re
import json
import typing as t
from pathlib import Path

import dateparser
from langchain_core.documents import Document


def canonicalize_value(name: t.Any) -> t.Any:
    if isinstance(name, str):
        return re.sub(r"[^a-zA-Z0-9]", "", name.lower())
    return name


def canonicalize_date(value: str) -> str:
    value = dateparser.parse(value)
    if value is not None:
        value = value.strftime("%Y-%m-%d")
    return value


def is_alt_text_img(s: str) -> bool:
    return bool(re.search(r"(?:----media/.+----|----Image alt text----.+)", s))


def validate_text(s: str) -> bool:
    return s.strip() and not is_alt_text_img(s)


def log_chat(
    log_file: Path,
    question: str,
    answer: str,
    retrieval_docs: list[Document] | None = None,
):
    if retrieval_docs:
        retrieval_docs = [doc.page_content for doc in retrieval_docs]
    else:
        retrieval_docs = []

    record = {
        "question": question,
        "answer": answer,
        "retrieval_docs": retrieval_docs,
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
