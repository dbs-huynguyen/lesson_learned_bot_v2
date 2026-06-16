import typing as t

from ragas.metrics import Faithfulness
from ragas.metrics._faithfulness import (
    StatementGeneratorInput,
    StatementGeneratorOutput,
    NLIStatementInput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
)
from langchain_core.callbacks import Callbacks

from metrics.base import PydanticPrompt


# class StatementGeneratorInput(BaseModel):
#     question: str = Field(..., description="The question to answer")
#     answer: str = Field(..., description="The answer to the question")


# class StatementGeneratorOutput(BaseModel):
#     statements: list[str] = Field(..., description="The generated statements")


class StatementGeneratorPrompt(PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]):
    instruction = "Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON."
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        )
    ]


# class StatementFaithfulnessAnswer(BaseModel):
#     statement: str = Field(..., description="The original statement, word-by-word")
#     reason: str = Field(..., description="The reason for the verdict")
#     verdict: int = Field(..., description="The verdict (0/1) of the faithfulness.")


# class NLIStatementInput(BaseModel):
#     context: str = Field(..., description="The context of the statements")
#     statements: list[str] = Field(..., description="The statements to judge")


# class NLIStatementOutput(BaseModel):
#     statements: list[StatementFaithfulnessAnswer]


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information given in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                statements=[
                    "Albert Einstein was a genius.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="The context and statement are unrelated",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


class MyFaithfulness(Faithfulness):
    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        text, question = row["response"], row["user_input"]

        statements = await self.statement_generator_prompt.generate(
            data=StatementGeneratorInput(question=question, answer=text),
            llm=self.llm,
            callbacks=callbacks,
            retries_left=0,
            temperature=0,
        )

        return statements

    async def _create_verdicts(
        self, row: dict, statements: list[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        verdicts = await self.nli_statements_prompt.generate(
            data=NLIStatementInput(
                context="\n\n".join(row["retrieved_contexts"]),
                statements=statements,
            ),
            llm=self.llm,
            callbacks=callbacks,
            retries_left=0,
            temperature=0,
        )

        return verdicts


def faithfulness(**kwargs) -> Faithfulness:
    kwargs["statement_generator_prompt"] = StatementGeneratorPrompt()
    kwargs["nli_statements_prompt"] = NLIStatementPrompt()

    return MyFaithfulness(**kwargs)
