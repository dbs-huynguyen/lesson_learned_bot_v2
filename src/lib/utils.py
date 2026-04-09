import os
import re
import pickle
import unicodedata
from enum import Enum

from rank_bm25 import BM25Okapi
from langchain_community.retrievers import BM25Retriever


VIETNAMESE_MAP = {
    "á": "a",
    "Á": "A",
    "à": "a",
    "À": "A",
    "ả": "a",
    "Ả": "A",
    "ã": "a",
    "Ã": "A",
    "ạ": "a",
    "Ạ": "A",
    "ă": "a",
    "Ă": "A",
    "ắ": "a",
    "Ắ": "A",
    "ằ": "a",
    "Ằ": "A",
    "ẳ": "a",
    "Ẳ": "A",
    "ẵ": "a",
    "Ẵ": "A",
    "ặ": "a",
    "Ặ": "A",
    "â": "a",
    "Â": "A",
    "ấ": "a",
    "Ấ": "A",
    "ầ": "a",
    "Ầ": "A",
    "ậ": "a",
    "Ậ": "A",
    "ẫ": "a",
    "Ẫ": "A",
    "é": "e",
    "É": "E",
    "è": "e",
    "È": "E",
    "ẻ": "e",
    "Ẻ": "E",
    "ẽ": "e",
    "Ẽ": "E",
    "ẹ": "e",
    "Ẹ": "E",
    "ê": "e",
    "Ê": "E",
    "ế": "e",
    "Ế": "E",
    "ề": "e",
    "Ề": "E",
    "ể": "e",
    "Ể": "E",
    "ễ": "e",
    "Ễ": "E",
    "ệ": "e",
    "Ệ": "E",
    "ó": "o",
    "Ó": "O",
    "ò": "o",
    "Ò": "O",
    "ỏ": "o",
    "Ỏ": "O",
    "õ": "o",
    "Õ": "O",
    "ọ": "o",
    "Ọ": "O",
    "ơ": "o",
    "Ơ": "O",
    "ớ": "o",
    "Ớ": "O",
    "ờ": "o",
    "Ờ": "O",
    "ở": "o",
    "Ở": "O",
    "ỡ": "o",
    "Ỡ": "O",
    "ợ": "o",
    "Ợ": "O",
    "ô": "o",
    "Ô": "O",
    "ố": "o",
    "Ố": "O",
    "ồ": "o",
    "Ồ": "O",
    "ổ": "o",
    "Ổ": "O",
    "ỗ": "o",
    "Ỗ": "O",
    "ộ": "o",
    "Ộ": "O",
    "ú": "u",
    "Ú": "U",
    "ù": "u",
    "Ù": "U",
    "ủ": "u",
    "Ủ": "U",
    "ũ": "u",
    "Ũ": "U",
    "ụ": "u",
    "Ụ": "U",
    "ư": "u",
    "Ư": "U",
    "ứ": "u",
    "Ứ": "U",
    "ừ": "u",
    "Ừ": "U",
    "ử": "u",
    "Ử": "U",
    "ữ": "u",
    "Ữ": "U",
    "ự": "u",
    "Ự": "U",
    "í": "i",
    "Í": "I",
    "ì": "i",
    "Ì": "I",
    "ỉ": "i",
    "Ỉ": "I",
    "ĩ": "i",
    "Ĩ": "I",
    "ị": "i",
    "Ị": "I",
    "ý": "y",
    "Ý": "Y",
    "ỳ": "y",
    "Ỳ": "Y",
    "ỷ": "y",
    "Ỷ": "Y",
    "ỹ": "y",
    "Ỹ": "Y",
    "ỵ": "y",
    "Ỵ": "Y",
    "đ": "d",
    "Đ": "D",
}


class TypeEnum(Enum):
    PROBLEM = "Problem Description"
    ROOT_CAUSE = "Root Cause"
    SOLUTION = "Solution"
    LESSON_LEARNED = "Lesson Learned"
    IMPROVEMENT = "Improvement"
    EVALUATION = "Evaluation"
    DATE = "Date"


class RoleEnum(Enum):
    REVIEWER = "Người xem xét"
    PERFORMER = "Người thực hiện"
    REPORTER = "Người báo cáo"


def to_snake_case(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for viet_char, replacement in VIETNAMESE_MAP.items():
        text = text.replace(viet_char, replacement)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def handle_link(match: re.Match[str], urls: dict[str, list[str]]) -> str:
    key = f"#{to_snake_case(match.group(2))}"
    urls[key] = [match.group(1), match.group(2)]
    return key


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"___+", "", s)
    s = s.lstrip("\n").rstrip()
    s = re.sub(r"^([IVXLCDM\da-z]\s*[\./\)])\s*(.+)", r"\1 \2", s)
    if re.search(r"-+media/image1.png-+", s):
        s = ""
        print(s)
    if match := re.match(r"^(\s*)--\s*(.+)", s):
        s = f"{match.group(1)}* {match.group(2)}"
    else:
        s = s.lstrip("\t").lstrip(" ")
        if re.search(r"^\u2610|\u2612", s):
            s = ""
        elif re.search(
            r"^[\da-z][\./\)] (?:Mô tả sự không phù hợp|Bài học kinh nghiệm|Đề xuất cập nhật rủi ro, cải tiến hệ thống(?: IMS)? \(nếu có\)|Đề xuất (?:cập nhật rủi ro|cải tiến)):?",
            s,
        ):
            s = ""
        elif match := re.match(
            r"^\d[\./\)] ((?:Đơn vị (?:có NC|xảy ra sự không phù hợp)|Từ khóa gán nhãn \(Key word/ Tag\))):?\s*(.+)?",
            s,
        ):
            s = f"* {match.group(1)}: {match.group(2)}" if match.group(2) else ""
        elif re.search(r"^Ngày", s):
            s = re.sub(r"\s+", " ", s)
    return s
