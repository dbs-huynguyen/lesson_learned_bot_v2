import os
import re
import json
import uuid
import datetime
import pprint
import inspect
import logging
import subprocess
from pathlib import Path
from typing import Any, Generator, Callable
from dotenv import load_dotenv

from docx2python import docx2python
from docx2python.depth_collector import Par
from docx2python.iterators import iter_paragraphs
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.utils.uuid import uuid7
from markdown_extract import MarkdownExtractor

from src.lib.utils import TypeEnum, RoleEnum, handle_link, clean_text
from src.lib.prompts import SUMMARIZE_REPORT_PROMPT


load_dotenv()
logger = logging.getLogger(__name__)


class MyDocument(Document):
    pass


class LessonsLearnedParserOld:

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
            raise ValueError(
                "File name does not match expected pattern: BM.10.2.01.BISO - Bao cao HDKP va BHKN - <created_date>.docx"
            )

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
                if match := re.match(r"^[\da-z][\./\)]\s*(.+)", p):
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

                if match := re.match(
                    r"^Ngày\s*(\d{2}\s*/\s*\d{2}\s*/\s*\d{4})", row[1][0]
                ):
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
                        text=f"{record['type']}\n{record['page_content']}",
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


IGNORE_HD_FILES = [
    "data/IMS/Hướng dẫn công việc/HD.7.1.3.A.HC - Su dung ghe nam y te.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.1.3.B.HC - Su dung và dat phong hop.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.1.5.2.BISO - Kiem tra cong cu do luong JMeter.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.1.6.PM - HDCV Viet cac huong dan lap trinh, danh gia ma nguon.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.1.HC - Tong hop Bao cao ngay.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.1A.HC - Dang ky nghi phep.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.1B.HC - Xac nhan quen cham cong.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.2A.HC-Ghi nhan va Gop y nang tam nhan vien.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.4.B.BISO - To chuc cuoc hop.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.4.B.HC-Don tiep khach hang.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.4.HC - Cong van DI - DEN.docx",
    "data/IMS/Hướng dẫn công việc/HD.7.4.PM - Cau noi.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.3.KD - Phat trien kinh doanh.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.4.2.PM - HDCV Chon framework, thu vien ngoai.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.5.1.HC - Kiem soat THU CHI TAI CHANH.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.5.1.PM - Kiem tra cac yeu cau phi chuc nang.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.5.1A.PM - HDCV Viet ma lenh de kiem thu tu dong.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.5.5.KD - Quan he khach hang.docx",
    "data/IMS/Hướng dẫn công việc/HD.8.5.5.PM - HDCV Ho tro van hanh phan mem.docx",
    "data/IMS/Hướng dẫn công việc/HD.9.3.BISO - Hop xem xet cua lanh dao.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.5.17.BISO - Quan ly mat khau an toan.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.5.23.BISO - Quản lý và sử dụng dịch vụ điện toán đám mây.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.5.35.BISO - Thu thap thong tin may tinh.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.7.1.HC - Don tiep khach trong cong ty.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.7.10.BISO - Chuong trinh KeePass de quan ly mat khau.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.7.14.HC - Giao nhan thiet bi.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.8.11.BISO - Lam mat na du lieu.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.7.8.BISO - Huong dan su dung may tinh an toan.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.8.19.BISO - Cai dat phan mem moi.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.8.20.BISO - Dang nhap VPN tren Windows.docx",
    "data/IMS/Hướng dẫn công việc/HD.A.9.2.1.BISO - Su dung thiet bi ca nhan truy cap HTTT cong ty va khach hang.docx",
    "data/IMS/Hướng dẫn công việc/HDCV Báo cáo ngày.docx",
]


IGNORE_BHKN_FILES = [
    "data/BHKN/AuthConsole_20231027.docx",
    # "data/BHKN/CDS_20250731.docx",
    # "data/BHKN/CDS_20250917.docx",
    # "data/BHKN/CDS_20260320.docx",
    "data/BHKN/KonnichinoAI_20260304.docx",
    "data/BHKN/Link_20250623.docx",
    "data/BHKN/MonshinApp_20251223.docx",
    # "data/BHKN/Pass_20241030.docx",
    # "data/BHKN/Pass_20260119.docx",
    "data/BHKN/Pass_20260122.docx",
    "data/BHKN/Pivot_20241021.docx",
    "data/BHKN/PreMonshinApp_20250716.docx",
]

class BaseParser:
    """BaseParser defines the interface and common functionality for parsing documents.

    Subclasses should implement the file_globs, and parser methods to specify how to find and parse documents.
    """
    def __init__(self, data_dir: Any = None) -> None:
        self._data_dir = data_dir

    @property
    def data_dir(self) -> Path:
        if not self._data_dir:
            raise ValueError("Data directory is not set")
        if isinstance(self._data_dir, str):
            self._data_dir = Path(self._data_dir)
        elif not isinstance(self._data_dir, Path):
            raise ValueError("Data directory must be a Path or str")
        return self._data_dir

    @property
    def allow_ext(self) -> set[str]:
        """Defines the allowed file extensions for document files.

        Returns:
            set[str]: A set of allowed file extensions (e.g., {".docx", ".pdf"}).
        """
        return {".docx"}

    @property
    def file_globs(self) -> set[str]:
        """Defines the glob patterns to use for finding document files.

        Returns:
            set[str]: A set of glob patterns to search for document files.
        """
        return {"*"}

    def convert_doc_to_docx(self, file_path: Path) -> Path:
        """Converts a .doc file to .docx format using docx2python.

        Args:
            file_path (Path): The path to the .doc file to convert.

        Returns:
            Path: The path to the converted .docx file.
        """
        if file_path.suffix != ".doc":
            raise ValueError("Input file must have a .doc extension")
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                f"{file_path.parent}",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(
            json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": result.stdout.strip().replace("\n", ". "),
                },
                ensure_ascii=False,
            )
        )
        if result.stderr:
            logger.error(
                json.dumps(
                    {
                        "event": inspect.currentframe().f_code.co_name,
                        "message": result.stderr.strip(),
                    },
                    ensure_ascii=False,
                )
            )
        return file_path.with_suffix(".docx")

    def filter_files(self) -> Generator[Path, None, None]:
        """Returns a generator of file paths that match the specified glob patterns and allowed extensions.
        
        Returns:
            Generator[Path, None, None]: A generator of file paths that match the criteria.
        """
        for glob in self.file_globs:
            for p in sorted(self.data_dir.rglob(glob)):
                if p.suffix in self.allow_ext and str(p) not in [*IGNORE_HD_FILES, *IGNORE_BHKN_FILES]:
                    yield p

    def parser(self, file_path: Path) -> Generator[list[MyDocument], None, None]:
        """Returns a generator of MyDocument objects containing the parsed data from the documents.

        Returns:
            Generator[MyDocument, None, None]: A generator of MyDocument objects containing the parsed data.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self):
        for p in self.filter_files():
            if ".doc" in self.allow_ext and p.suffix == ".doc":
                p = self.convert_doc_to_docx(p)
            yield from self.parser(p)


def is_alt_text_img(s: str) -> bool:
    return bool(re.search(r"(?:----media/.+----|----Image alt text----.+)", s))


def validate_text(s: str) -> bool:
    return s.strip() and not is_alt_text_img(s)


class LessonsLearnedParser(BaseParser):
    def __init__(self, data_dir: Any = None) -> None:
        super().__init__(data_dir=data_dir)
        self._summarize_model = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_LLM_MODEL"),
            validate_model_on_init=False,
            client_kwargs={"timeout": 120},
            temperature=0.7,
            reasoning=False,
            num_ctx=4096,
            num_predict=1024,
            seed=9999,
        )

    @property
    def file_globs(self) -> set[str]:
        return {"BHKN/**/*.docx"}

    @property
    def allow_ext(self) -> set[str]:
        return {".docx", ".doc"}

    def _get_title(self, file_path: Path) -> str:
        docx = docx2python(file_path, duplicate_merged_cells=False)
        return "".join(
            [
                "".join([ele.strip().replace("\n", " ") for ele in p.run_strings])
                for p in iter_paragraphs(docx.header_pars)
                if p.style == "Title"
            ]
        ).upper()

    def _get_body(self, file_path: Path) -> tuple[str, str | None]:
        docx = docx2python(file_path, duplicate_merged_cells=False)
        occurred_at = None
        merged: list[str] = []
        table_header: list[Par] = []
        table_body: list[Par] = []
        for p in iter_paragraphs(docx.body_pars):
            if not p.style:
                continue

            if len(table_header) > 0 and p.style not in ["TableHeader", "TableBody"]:
                header: list[str] = ["".join([p.strip().replace("\n", " ") for p in row.run_strings]) for row in table_header]
                spliter: list[str] = ["---" for _ in range(len(table_header))]

                tbl_md = f"| {' | '.join(header)} |\n"
                tbl_md += f"| {' | '.join(spliter)} |\n"
                for i in range(0, len(table_body), len(table_header)):
                    for row in table_body[i : i + len(table_header)]:
                        tbl_md += f"| {' | '.join([p.strip().replace("\n", "<br>") for p in row.run_strings])} |\n"

                merged.append(tbl_md)
                table_header.clear()
                table_body.clear()

            # print(f"{p.style} - {p.lineage}: {p.run_strings}")

            if p.style == "TableHeader":
                table_header.append(p)
                continue

            if p.style == "TableBody":
                table_body.append(p)
                continue


            prefix = ""
            if p.style == "Heading1":
                prefix = "# "
            elif p.style == "Heading2":
                prefix = "## "
            elif p.style == "Heading3":
                prefix = "### "
            elif p.style == "Heading4":
                prefix = "#### "
            elif p.style == "Heading5":
                prefix = "##### "
            elif p.style == "Heading6":
                prefix = "###### "
            elif p.style == "ListBullet1":
                prefix = ""
            elif p.style == "ListBullet2":
                prefix = "  "
            elif p.style == "ListBullet3":
                prefix = "    "
            elif p.style == "ListBullet4":
                prefix = "      "
            elif p.style == "ListBullet5":
                prefix = "        "
            elif p.style == "ListBullet6":
                prefix = "          "
            elif p.style == "OccurredDate":
                matched = re.match(r".+(\d{2})\/(\d{2})\/(\d{4})", p.run_strings[0].replace(" ", ""))
                if matched and len(matched.groups()) == 3:
                    occurred_at = datetime.datetime(*map(int, matched.groups()[::-1])).isoformat()

            text = " ".join([ele for ele in p.run_strings if validate_text(ele)]).strip()

            if text:
                text = re.sub(r"\t", "", text)
                text = re.sub(r"\s\s+", " ", text)
                text = re.sub(r"^--", "*", text)
                # Replace characters belonging to the Private Use Area with the • character to avoid errors during storage and display.
                text = re.sub(r"^[\uE000-\uF8FF]", "•", text)
                merged.append(f"{prefix}{text}")

        body_text = "\n\n".join(merged)
        return body_text, occurred_at

    def _summarize_content(self, document: str) -> str:
        prompt = SUMMARIZE_REPORT_PROMPT.format(document=document)
        return self._summarize_model.invoke(prompt)

    def parser(self, file_path: Path) -> Generator[list[MyDocument], None, None]:
        logger.info(
            json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": f"Parsing {file_path}",
                },
                ensure_ascii=False,
            )
        )

        body_text, occurred_at = self._get_body(file_path)
        summary = self._summarize_content(body_text)
        # print(summary)

        extractor = MarkdownExtractor(summary)
        
        docs: list[MyDocument] = []
        for page_number, section in enumerate(extractor.list(), 1):
            id = str(uuid7())
            docs.append(
                MyDocument(
                    id=id,
                    page_content=extractor.get_section(section).strip(),
                    metadata=dict(
                        doc_id=id,
                        source=file_path.name,
                        page_number=page_number,
                        section=section,
                        doc_type="BHKN",
                        occurred_at=occurred_at,
                    ),
                )
            )

        if len(docs) == 0:
            raise ValueError(f"No valid content extracted from {file_path}")

        yield docs


class WorkInstructionParser(BaseParser):
    @property
    def file_globs(self) -> set[str]:
        return {"IMS/Hướng dẫn công việc/*.docx"}

    @property
    def allow_ext(self) -> set[str]:
        return {".docx", ".doc"}

    def _get_title(self, file_path: Path) -> str:
        docx = docx2python(file_path, duplicate_merged_cells=False)
        return "".join(
            [
                "".join(p.run_strings)
                for p in iter_paragraphs(docx.header_pars)
                if p.style == "Title"
            ]
        ).upper()

    def _get_body(self, file_path: Path) -> str:
        docx = docx2python(file_path, duplicate_merged_cells=True)
        merged = []
        table: dict[str, list[Par]] = {"headers": [], "body": []}
        for p in iter_paragraphs(docx.body_pars):
            if not p.style:
                continue

            if len(table["headers"]) > 0 and p.style not in ["TableHeader", "TableBody"]:
                tbl_md = (
                    " | "
                    + " | ".join(
                        [
                            "".join(
                                [
                                    p.strip().replace("\n", "<br>")
                                    for p in row.run_strings
                                ]
                            )
                            for row in table["headers"]
                        ]
                    )
                    + " | "
                    + "\n"
                )
                tbl_md += (
                    " | " + " | ".join(["---" for _ in range(len(table["headers"]))]) + " | " + "\n"
                )
                tbl_md += "\n".join(
                    [
                        " | "
                        + " | ".join(
                            [
                                "".join(
                                    [
                                        p.strip().replace("\n", "<br>")
                                        for p in row.run_strings
                                    ]
                                )
                                for row in table["body"][i : i + len(table["headers"])]
                            ]
                        )
                        + " | "
                        for i in range(0, len(table["body"]), len(table["headers"]))
                    ]
                )
                merged.append(tbl_md)
                table["headers"].clear()
                table["body"].clear()

            # print(f"{p.style} - {p.lineage}: {p.run_strings}")

            if p.style == "TableHeader":
                table["headers"].append(p)
            elif p.style == "TableBody":
                table["body"].append(p)
            else:
                if p.style == "Heading1":
                    prefix = "# "
                elif p.style == "Heading2":
                    prefix = "## "
                elif p.style == "Heading3":
                    prefix = "### "
                elif p.style == "Heading4":
                    prefix = "#### "
                elif p.style == "Heading5":
                    prefix = "##### "
                elif p.style == "Heading6":
                    prefix = "###### "
                elif p.style == "ListBullet1":
                    prefix = ""
                elif p.style == "ListBullet2":
                    prefix = "  "
                elif p.style == "ListBullet3":
                    prefix = "    "
                elif p.style == "ListBullet4":
                    prefix = "      "
                elif p.style == "ListBullet5":
                    prefix = "        "
                elif p.style == "ListBullet6":
                    prefix = "          "
                else:
                    prefix = ""

                text = " ".join(
                    [ele for ele in p.run_strings if validate_text(ele)]
                ).strip()

                if text:
                    text = re.sub(r"\t", "", text)
                    text = re.sub(r"\s\s+", " ", text)
                    text = re.sub(r"^--", "*", text)
                    # Replace characters belonging to the Private Use Area with the • character to avoid errors during storage and display.
                    text = re.sub(r"^[\uE000-\uF8FF]", "•", text)
                    merged.append(f"{prefix}{text}")

        body_text = "\n\n".join(merged)
        return body_text

    def parser(self, file_path: Path) -> Generator[MyDocument, None, None]:
        logger.info(
            json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": f"Parsing file: {file_path}",
                },
                ensure_ascii=False,
            )
        )

        title = self._get_title(file_path)
        print(title)

        body_text = self._get_body(file_path)
        print(body_text)

        print("=" * 100)

        yield MyDocument(
            id=str(uuid.uuid4()),
            text=body_text,
            metadata=dict(
                source=file_path.name,
                title=title,
                type="work_instruction",
            ),
        )


class ProcessProcedureParser(BaseParser):
    pass


class ApplicableRegulationParser(BaseParser):
    pass


class PolicyParser(BaseParser):
    pass


class GoalParser(BaseParser):
    pass


class JobDescriptionParser(BaseParser):
    pass


class ResponsibilityAuthorityParser(BaseParser):
    pass


class HandbookParser(BaseParser):
    pass
