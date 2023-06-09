# MNLPs
This repository contains a Python module that implements natural language processing using GPT-3 and GPT-3.5 models in functions for deep and accurate text analysis, as well as generating small building blocks for use in more complex systems.

## MNLPs   
The module explores the use of GPTs in Mini Natural Language Processors (MNLPs) to perform text analysis tasks with a high degree of accuracy and depth.

## Dependencies
- pydash
- typing
- openai
- pydantic

## Usage
The module includes a bool_mnlp function that uses the OpenAI API to generate natural language responses to given statements and text. To use the function, import it from the module:

python
``
from bool_mnlp import bool_mnlp
``
Then call the function with a statement and a payload argument, where statement is a string containing the statement to be evaluated by the model and payload is a string containing text to be used as context for the evaluation:

python
``
result = bool_mnlp("statement to be evaluated", "text used as context")
``
The function returns a boolean value (True if the model responds affirmatively, False if it responds negatively).

## CompletionParams class
The module also includes a CompletionParams class that is used to specify the parameters for generating completions with the OpenAI API. The generate_completion function in the module takes a CompletionParams object as input and returns the response from the OpenAI API as an OpenAIObject.

## License
This code is released under the MIT License.

## About
This repository is based on the article "GPTs en Mini Procesadores del lenguaje natural A.K.A MNLPs (Spanish Version)", which explores the implementation of GPT-3 and GPT-3.5 models in natural language processing tasks.
