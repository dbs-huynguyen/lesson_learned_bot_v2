import re
import typing as t

import dateparser


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
