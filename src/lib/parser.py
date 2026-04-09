import re
import uuid
from pathlib import Path
from typing import Any, Generator, TypedDict
from collections.abc import Callable

from docx2python import docx2python

from src.lib.utils import TypeEnum, RoleEnum, handle_link, clean_text


class MyDocument(TypedDict):
    id: str
    text: str
    metadata: dict[str, Any]


class LessonsLearnedParser:

    def get_table(
        self,
        body: list[list[list[list[str]]]],
        clean_fn: Callable[[str], str] = lambda s: s,
    ) -> list[list[list[list[str]]]]:
        if not (
            isinstance(body, list)
            and isinstance(body[0], list)
            and isinstance(body[0][0], list)
            and isinstance(body[0][0][0], list)
            and isinstance(body[0][0][0][0], str)
        ):
            raise ValueError("Input must be a 4D list of strings")

        body[:] = [
            [
                row
                for raw_row in table
                if (
                    row := [
                        [cleaned for s in c if (cleaned := clean_fn(s))]
                        for c in raw_row
                    ]
                )
            ]
            for table in body
            if all(len(row) >= 2 for row in table)
        ]

        return body

    def transform(
        self, table: list[list[list[str]]], file_path: Path
    ) -> list[dict[str, Any]]:
        if not (
            isinstance(table[0], list)
            and isinstance(table[0][0], list)
            and isinstance(table[0][0][0], str)
        ):
            raise ValueError("Input must be a 3D list of strings")

        created_date = None, None, None
        if match := re.match(
            r"^BM.10.2.01.BISO - Bao cao HDKP va BHKN - (\S+)", file_path.stem
        ):
            created_date = re.sub(r"(\d{4})(\d{2})(\d{2})", r"\3/\2/\1", match.group(1))

        if not (created_date):
            raise ValueError("File name does not match expected pattern: BM.10.2.01.BISO - Bao cao HDKP va BHKN - <created_date>.docx")

        results: list[dict[str, Any]] = []
        for row in table:
            if len(row[0]) == 0:
                continue

            obj = {
                "created_date": created_date,
                "source": file_path.name,
            }

            heading1 = row[0].pop(0)
            if re.search(r"^I[\./\)]", heading1):
                obj["type"] = TypeEnum.PROBLEM.value
            elif re.search(r"^II[\./\)]", heading1):
                obj["type"] = TypeEnum.ROOT_CAUSE.value
            elif re.search(r"^III[\./\)]", heading1):
                obj["type"] = TypeEnum.SOLUTION.value
            elif re.search(r"^IV[\./\)]", heading1):
                obj["type"] = TypeEnum.LESSON_LEARNED.value
            elif re.search(r"^V[\./\)]", heading1):
                obj["type"] = TypeEnum.EVALUATION.value

            sentences = []
            ori_sentences = []
            urls = {}
            for i, p in enumerate(row[0][::-1]):
                if (not ori_sentences and re.search(r"^[\da-z][\./\)]", p)) or (
                    ori_sentences
                    and re.search(r"^[\da-z][\./\)]", ori_sentences[-1])
                    and re.search(r"^[\da-z][\./\)]", p)
                ):
                    continue

                ori_p = p
                if (match := re.match(r"^[\da-z][\./\)]\s*(.+)", p)):
                    prefix = ""
                    if i + 1 < len(row[0]):
                        prefix = "\n"
                    p = f"{prefix}{match.group(1)}"

                if re.search(r"""<a[^>]*.+</a>""", p):
                    p = re.sub(
                        r"""<a[^>]*href=["'](.*?)["'][^>]*>(.*?)</a>""",
                        lambda m: handle_link(m, urls),
                        p,
                    )

                if re.search(r"\d{4}[/\-]\d{2}[/\-]\d{2}", p):
                    p = re.sub(r"(\d{4})[/\-](\d{2})[/\-](\d{2})", r"\3/\2/\1", p)

                sentences.append(f"{p}  ")
                ori_sentences.append(ori_p)

            updated_at, role, owner = None, None, None
            if len(row) == 2 and len(row[1]) > 0:

                if match := re.match(r"^Ngày\s*(\d{2}\s*/\s*\d{2}\s*/\s*\d{4})", row[1][0]):
                    updated_at = match.group(1).replace(" ", "")

                if re.search(rf"{RoleEnum.REVIEWER.value}", row[1][1]):
                    role = RoleEnum.REVIEWER.name.lower()
                elif re.search(rf"{RoleEnum.PERFORMER.value}", row[1][1]):
                    role = RoleEnum.PERFORMER.name.lower()
                elif re.search(rf"{RoleEnum.REPORTER.value}", row[1][1]):
                    role = RoleEnum.REPORTER.name.lower()

                if len(row[1]) == 3:
                    owner = row[1][2].replace("\n", ", ")

            obj["page_content"] = "\n".join(sentences[::-1])
            obj["urls"] = urls or None
            obj["updated_at"] = updated_at
            obj["role"] = role
            obj["editor"] = owner

            results.append(obj)

        return results

    def parser(self, file_paths: list[Path]) -> Generator[MyDocument, Any, None]:
        for file_path in file_paths:
            with docx2python(file_path, duplicate_merged_cells=False) as docx:
                cleaned_body = self.get_table(docx.body, clean_text)
                raw_records = self.transform(cleaned_body[0], file_path=file_path)

                for record in raw_records:
                    if not record["page_content"]:
                        continue
                    yield MyDocument(
                        id=str(uuid.uuid4()),
                        text=(
                            f"""{record["type"]}\n\n"""
                            f"""{record["page_content"]}"""
                        ),
                        metadata=dict(
                            created_date=record["created_date"],
                            source=record["source"],
                            type=record["type"],
                            urls=record["urls"],
                            updated_at=record["updated_at"],
                            role=record["role"],
                            editor=record["editor"],
                        ),
                    )
