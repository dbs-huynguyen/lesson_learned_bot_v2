from pydantic import BaseModel, Field
from ragas.metrics import AnswerRelevancy

from metrics.base import PydanticPrompt


class ResponseRelevanceInput(BaseModel):
    response: str = Field(..., description="The answer to the question")


class ResponseRelevanceOutput(BaseModel):
    question: str = Field(..., description="The question to answer")
    noncommittal: int = Field(..., description="Whether the answer is noncommittal (1 for Yes, 0 for No)")


class ResponseRelevancePrompt(PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]):
    instruction = (
        "Generate a question for the given answer and identify if answer is noncommittal. "
        "Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. "
        "A noncommittal answer is one that is evasive, vague, or ambiguous. "
        "For example, \"I don't know\" or \"I'm not sure\" are noncommittal answers.\n"
    )
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="Albert Einstein was born in Germany.",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022.",
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


def answer_relevancy(**kwargs) -> AnswerRelevancy:
    kwargs["question_generation"] = ResponseRelevancePrompt()
    return AnswerRelevancy(**kwargs)
