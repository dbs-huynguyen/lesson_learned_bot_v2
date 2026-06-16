import json
import typing as t

from ragas.prompt import PydanticPrompt as RPydanticPrompt, InputModel, OutputModel


class PydanticPrompt(RPydanticPrompt[InputModel, OutputModel]):
    def _generate_output_signature(self, indent: int = 4) -> str:
        return (
            f"Complies with the following schema as specified in JSON Schema:\n"
            f"{json.dumps(self.output_model.model_json_schema(), indent=indent)}\n"
            "\n--------JSON Format Instructions-----------\n"
            "1. Only use the fields defined in the schema.\n"
            "2. Only use compatible operators.\n"
            "3. Maintain the exact format as defined with enum values and nested structures.\n"
            "4. Do not use single quotes but double quotes, properly escaped with a backslash.\n"
            "5. Returns valid JSON."
        )

    def _generate_examples(self):
        if self.examples:
            example_strings = []
            for idx, e in enumerate(self.examples):
                input_data, output_data = e
                example_strings.append(
                    f"Example {idx + 1}\n"
                    + "Input: "
                    + input_data.model_dump_json(indent=4)
                    + "\n"
                    + "Output: "
                    + output_data.model_dump_json(indent=4)
                )

            return "\n--------EXAMPLES-----------\n" + "\n\n".join(example_strings)
        # if no examples are provided
        else:
            return ""

    def to_string(self, data: t.Optional[InputModel] = None) -> str:
        return (
            f"{self.instruction}\n"
            + self._generate_output_signature()
            + "\n-----------------------------\n"
            + self._generate_examples()
            + "\n-----------------------------\n"
            + "\nNow perform the same with the following input\n"
            + (
                "input: " + data.model_dump_json(indent=4, exclude_none=True) + "\n"
                if data is not None
                else "Input: (None)\n"
            )
            + "Output: "
        )
