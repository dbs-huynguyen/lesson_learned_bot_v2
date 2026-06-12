from dotenv import load_dotenv

load_dotenv()

import re
import json
import uuid
import datetime
import inspect
import logging
import subprocess
import typing as t
from pathlib import Path

from docx2python import docx2python
from docx2python.depth_collector import Par
from docx2python.iterators import iter_paragraphs
from langchain_core.documents import Document
from langchain_core.utils.uuid import uuid7
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.agent.common import get_base_llm
from src.lib.utils import validate_text
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

logger = logging.getLogger(__name__)


ADD_CONTEXTUAL_PROMPT = ChatPromptTemplate.from_template("""
Bạn là chuyên gia trong việc bổ sung ngữ cảnh cho các đoạn văn bản.

Cho một tài liệu markdown mô tả sự cố, hãy mô tả lại sự cố bằng một đoạn văn ngắn (1-2 câu).

Đoạn văn cần bao gồm:
- Các thực thể chính được đề cập trong tài liệu mô tả sự cố.

Không thêm các từ dẫn như: 'Sự cố xảy ra khi', 'Sự cố', 'Sự không phù hợp',...
Chỉ xuất ra tiền tố ngữ cảnh, không có gì khác.

Tài liệu mô tả sự cố:
{document}
""".strip())


EXTRACT_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một kỹ sư Dữ liệu Cấp cao và chuyên gia QA Phần mềm.
Nhiệm vụ của bạn là chuyển đổi tài liệu Incident/CAPA phần mềm dạng Markdown thành cấu trúc JSON đề phục vụ hệ thống RAG nâng cao.

JSON schema bắt buộc:
{schema}

Yêu cầu:
- Chỉ trả về JSON hợp lệ, không giải thích gì thêm.

Tài liệu gốc:
{markdown_text}
""".strip())


GENERATE_RELEVANT_QUESTIONS_SYSTEM_PROMPT = PromptTemplate(
    template="""
Bạn là một kỹ sư Dữ liệu Cấp cao và chuyên gia QA Phần mềm.
Nhiệm vụ của bạn là chuyển đổi tài liệu Incident/CAPA phần mềm dạng Markdown thành cấu trúc JSON đề phục vụ hệ thống RAG nâng cao.

Quyền lực đặc biệt dành cho bạn (Để tối ưu Score cho câu hỏi Tổng hợp):
Hệ thống RAG hiện tại đang gặp lỗi: Khi người dùng hỏi câu chi tiết kỹ thuật thì tìm được, nhưng hỏi câu tổng hợp như 'Hãy tổng hợp các biện pháp phòng ngừa...' hoặc 'Bài học kinh nghiệm về hệ thống X' thì Score Embedding bị thấp do lệch pha từ khóa.

Vì vậy, tại trường 'cau_hoi', bạn hãy đóng vai một Giám đốc Công nghệ hoặc Kiểm toán viên QA, tự suy nghĩ ra 3-5 câu hỏi mang tính KHÁI QUÁT, TỔNG HỢP hoặc PHÒNG NGỪA RỦI RO hướng đến các giải pháp/bài học trong tài liệu này. 
Các câu hỏi này PHẢI chứa các cụm từ như: 'tổng hợp', 'biện pháp phòng ngừa', 'ngăn ngừa phát sinh', 'bài học kinh nghiệm', 'cải tiến quy trình'.

JSON schema bắt buộc:
{schema}

Yêu cầu:
- Chỉ trả về JSON hợp lệ, không giải thích gì thêm.

Tài liệu:
{context}
""".strip(),
    input_variables=["context", "schema"],
)


class MyDocument(Document):
    pass


CHUNK_TYPE = {
    1: "mo_ta",
    2: "nguyen_nhan",
    3: "khac_phuc",
    4: "bai_hoc",
}


TOPIC = {
    1: "Mô tả chi tiết",
    2: "Nguyên nhân gốc",
    3: "Biện pháp khắc phục",
    4: "Bài học rút ra",
}


class BaseParser:
    def __init__(self, data_dir: t.Any = None) -> None:
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

    def filter_files(self) -> t.Generator[Path, None, None]:
        """Returns a generator of file paths that match the specified glob patterns and allowed extensions.

        Returns:
            Generator[Path, None, None]: A generator of file paths that match the criteria.
        """
        for glob in self.file_globs:
            for p in sorted(self.data_dir.rglob(glob)):
                if p.suffix in self.allow_ext:
                    yield p

    def parser(self, file_path: Path) -> t.Generator[list[MyDocument], None, None]:
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


class LessonsLearnedParser(BaseParser):
    def __init__(self, data_dir: t.Any = None) -> None:
        super().__init__(data_dir=data_dir)

        self._add_contextual_chain = self._add_contextual()

    @property
    def file_globs(self) -> set[str]:
        return {"BHKN/**/*.docx", "BHKN/**/*.doc"}

    @property
    def allow_ext(self) -> set[str]:
        return {".docx", ".doc"}

    def _get_body(self, file_path: Path) -> tuple[str, str | None, str | None]:
        docx = docx2python(file_path, duplicate_merged_cells=False)
        occurred_at = None
        department = None
        merged: list[str] = []
        table_header: list[Par] = []
        table_body: list[Par] = []
        for p in iter_paragraphs(docx.body_pars):
            if not p.style or p.style in ["Normal", "Title"]:
                continue

            if len(table_header) > 0 and p.style not in ["TableHeader", "TableBody"]:
                header: list[str] = [
                    "".join([p.strip().replace("\n", " ") for p in row.run_strings])
                    for row in table_header
                ]
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

            text = " ".join(
                [ele for ele in p.run_strings if validate_text(ele)]
            ).strip()

            prefix = ""
            if p.style == "Heading1":
                prefix = "# "
                text = re.sub(r"(?:[IVX]+|\d+|[A-Z])\)\s*", "", text)
            elif p.style == "Heading2":
                prefix = "## "
                text = re.sub(r"(?:[IVX]+|\d+|[A-Z])\)\s*", "", text)
            elif p.style == "Heading3":
                prefix = "### "
                text = re.sub(r"(?:[IVX]+|\d+|[A-Z])\)\s*", "", text)
            elif p.style == "Heading4":
                prefix = "#### "
                text = re.sub(r"(?:[IVX]+|\d+|[A-Z])\)\s*", "", text)
            elif p.style == "Heading5":
                prefix = "##### "
                text = re.sub(r"(?:[IVX]+|\d+|[A-Z])\)\s*", "", text)
            elif p.style == "Heading6":
                prefix = "###### "
                text = re.sub(r"(?:[IVX]+|\d+|[A-Z])\)\s*", "", text)
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
                matched = re.match(
                    r".+(\d{2})\/(\d{2})\/(\d{4})", text.replace(" ", "")
                )
                if matched and len(matched.groups()) == 3:
                    occurred_at = datetime.datetime(
                        *map(int, matched.groups()[::-1])
                    ).strftime("%Y-%m-%d")
                continue
            elif p.style == "Department":
                department = text

            if text:
                text = re.sub(r"\t", "", text)
                text = re.sub(r"\s\s+", " ", text)
                text = re.sub(r"^--", "*", text)
                # Replace characters belonging to the Private Use Area with the • character to avoid errors during storage and display.
                text = re.sub(r"^[\uE000-\uF8FF]", "•", text)
                merged.append(f"{prefix}{text}")

        body_text = "\n\n".join(merged)
        return body_text, occurred_at, department

    def _add_contextual(self):
        return (
            ADD_CONTEXTUAL_PROMPT
            | get_base_llm(
                top_p=0.9,
                repeat_penalty=1,
                presence_penalty=0,
                temperature=0.3,
            )
            | StrOutputParser()
        )

    def parser(
        self, file_path: Path
    ) -> t.Generator[list[MyDocument], None, None]:
        logger.info(
            json.dumps(
                {
                    "event": inspect.currentframe().f_code.co_name,
                    "message": f"Parsing {file_path}",
                },
                ensure_ascii=False,
            )
        )

        matched = re.match(r"^([a-zA-Z0-9]+)", file_path.stem)
        if not matched:
            raise ValueError(
                f"File name does not match expected pattern: <project_name>_yyyymmdd, got {file_path.name}"
            )

        project_name = matched.group(1).lower()

        if file_path.suffix == ".docx":
            body_text, occurred_at, department = self._get_body(file_path)
            with open(file_path.with_suffix(".md"), "w", encoding="utf-8") as f:
                f.write(body_text)
        elif file_path.suffix == ".md":
            body_text = file_path.read_text(encoding="utf-8")
            occurred_at, department = None, None
        else:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "section")], strip_headers=True
        )
        chunks = splitter.split_text(body_text)
        print(len(chunks))

        docs: list[MyDocument] = []

        contextual_prefix = self._add_contextual_chain.invoke(
            chunks[0].page_content
        ).strip()

        for i, chunk in enumerate(chunks, 1):
            section = chunk.metadata.get("section") or ""
            if i in [1, 5]:
                print(f"Skipping chunk {section}")
                continue

            chunk_id = str(uuid7())
            content = f"### Mô tả sự cố:\n{contextual_prefix}\n### {TOPIC[i]}:\n{chunk.page_content}"
            print(content)
            print("-" * 80)
            docs.append(
                MyDocument(
                    id=chunk_id,
                    page_content=content,
                    metadata=dict(
                        doc_type="BHKN",
                        source=file_path.name,
                        occurred_at=occurred_at,
                        project_name=project_name,
                        department=department,
                        chunk_type=CHUNK_TYPE[i],
                        page_number=i,
                    ),
                )
            )

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

            if len(table["headers"]) > 0 and p.style not in [
                "TableHeader",
                "TableBody",
            ]:
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
                    " | "
                    + " | ".join(["---" for _ in range(len(table["headers"]))])
                    + " | "
                    + "\n"
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

    def parser(self, file_path: Path) -> t.Generator[MyDocument, None, None]:
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
        # print(title)

        body_text = self._get_body(file_path)
        # print(body_text)

        # print("=" * 100)

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
