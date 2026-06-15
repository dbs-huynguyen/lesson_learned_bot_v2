import json

from ragas.prompt import PydanticPrompt as RPydanticPrompt, InputModel, OutputModel


class PydanticPrompt(RPydanticPrompt[InputModel, OutputModel]):
    def _generate_output_signature(self, indent: int = 4) -> str:
        return (
            f"Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:\n"
            f"{json.dumps(self.output_model.model_json_schema(), indent=indent)}\n"
            "Do not use single quotes in your response but double quotes, properly escaped with a backslash."
        )
