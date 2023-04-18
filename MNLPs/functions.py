from pydash import get
from typing import List, Optional, Union, cast
import openai
from pydantic import BaseModel
from openai.openai_object import OpenAIObject

openai.api_key = "your-api-key"


class CompletionParams(BaseModel):
    model: str
    prompt: Optional[Union[str, List[Union[str, List[str]]]]] = None
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    best_of: Optional[int] = 1
    logit_bias: Optional[dict] = None
    user: Optional[str] = None


def generate_completition(input: CompletionParams) -> OpenAIObject:
    return openai.Completion.create(**input.dict(exclude_none=True))


def bool_mnlp(statement: str, paylaod: str):
    """
    This function uses the OpenAI API to generate a natural language response to a given
    statement and text, and returns a boolean value based on the response. The response
    is generated using the text-davinci-003 model with a temperature of 0 and max_tokens of 2.

    Parameters:
    - statement: a string containing a statement to be evaluated by the model.
    - paylaod: a string containing text to be used as context for the evaluation.

    Returns:
    - A boolean value, True if the model responds affirmatively (SI), False if it responds
    negatively (NO).

    Raises:
    - TypeError: if the completion text is not found in the OpenAI response or has an
    unexpected value.
    """
    rule = (
        "'SI' SI ES AFIRMATIVO O 'NO' SI ES NEGATIVO EL STATEMENT DEL TEXTO, SI NO 'NO'"
    )
    statement = f"STATEMENT:{statement}"
    text = f"TEXTO:{paylaod}"

    prompt = f"{rule}\n{statement}\n{text}\nRESPUESTA:"
    completition = generate_completition(
        input=CompletionParams(
            model="text-davinci-003",
            temperature=0,
            prompt=prompt,
            max_tokens=2,
        )
    )
    completition_text = get(completition, "choices.[0].text", default=None)
    completition_text = cast(Optional[str], completition_text)
    if completition_text is None:
        raise TypeError("Text not found on completition")

    completition_text = completition_text.upper()
    if completition_text.find("NO") != -1:
        return False
    elif completition_text.find("SI") != -1:
        return True
    else:
        raise TypeError(f"Bad completiton text value {completition_text}")
