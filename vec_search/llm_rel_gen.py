"""
This module houses the framework for generating LLM relevances for search queries.
"""
import os

from abc import ABC
from pathlib import Path
from string import Template
from typing import Union

# llm clients
from openai import OpenAI
from vertexai.generative_models import GenerationConfig, GenerativeModel

class LLMRelAssessorBase(ABC):
    def __init__(
        self,
        prompt: Union[str, Template] = "",
        model_name: str = "",
        shot_count: int = 0
    ):
        self.shot_count = shot_count
        self.model_name = model_name
        self.prompt = prompt if Template else Template(prompt)

    def _create_client(self):
        raise NotImplementedError("error: you must implement this method in a child class ...")


class LLMRelAssessor(LLMRelAssessorBase):
    def __init__(
        self,
        # qrel,
        prompt: Union[str, Template] = "",
        model_name: str = "openai",
        shot_count: int = 0
    ):
        super().__init__(shot_count=shot_count, prompt=prompt, model_name=model_name)
        # TODO: consider using `dotenv` package as part of app config overal refactor
        # if not os.environ["OPEN_AI_API_KEY"]:
        #    raise ValueError("error the OPEN_AI_API_KEY environment variable is not set")
        # self.qrel = qrel # TODO: figure out why this is necessary in trec & umbrela codes ...
        self._create_client(model_name=model_name)

    def _create_client(self, model_name: str) -> None:
        if model_name == "openai":
            self.model_name = "gpt-4o-mini"
            api_key = os.environ["OPEN_AI_API_KEY"]
            self.client = OpenAI(api_key=api_key)
        elif model_name == "gemini":
            # requires a project setup in google cloud-make env
            PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
            LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
            import vertexai
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            gen_config = GenerationConfig(temperature=0.1, max_output_tokens=256)
            # NOTE: 1.5 has stricter api limits but models response format varies a lot
            # TODO: figure out batch api calls if supported
            # model = "gemini-1.0-pro" # does not comport well with prompt
            model = "gemini-1.5-pro"
            gen_model = GenerativeModel(model_name=model, generation_config=gen_config)
            self.client = gen_model
        else:
            raise ValueError(f"model {self.model_name} is not currently supported ...")

    def generate_rel(self, template_params: Union[None, dict[str, str]] = None, parse: bool = True):
        # for now assume openai client and ref:
        # https://platform.openai.com/docs/api-reference/debugging-requests
        # for error/debug codes etc.
        content = self.prompt.safe_substitute(template_params)
        # !!! TODO !!!: `ABC.this` the difficulty in interface is actually clients ...
        # basically  we want an common interface for llm clients and have this
        # class `self` use that interface...
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": content,
                }],
                model=self.model_name,
            )
        elif isinstance(self.client, GenerativeModel):
            response = self.client.generate_content(contents=content)
            # breakpoint()
        else:
            raise NotImplementedError(f"generate_rel for client: {self.client!r} not implemented...")
        return self.parse_resp(response) if parse else response

    def parse_resp(self, resp) -> dict[str, str]:
        # openai docs on response object
        # https://platform.openai.com/docs/api-reference/chat/object
        # NOTE: maybe type this as a 'ParsedResponse'...
        stuff = {
            'llm_client': '',
            'msg-id': '',
            'model-id': '',
            'finish-reason': '',
            'message': '',
            'usage': ''
        }
        # both OpenAI and vertexai.GenerativeModel have the .to_dict() method
        rd = resp.to_dict()
        if isinstance(self.client, OpenAI):
            if len(rd['choices']) > 1:
                print("multiple choices in response, taking the first...")
            choice = rd['choices'][0]
            # stuff ...
            stuff['msg-id'] = rd['id']
            stuff['model-id'] = rd['model']
            stuff['usage'] = rd['usage']
            stuff['message'] = choice['message']['content']
            stuff['finish-reason'] = choice['finish_reason']
            # ...
            llm_client = 'openai'
        elif isinstance(self.client, GenerativeModel):
            if len(rd['candidates']) > 1:
                print("multiple candidates in response, taking the first...")
            choice = rd['candidates'][0]['content']
            stuff['msg-id'] = None
            stuff['model-id'] =  rd['model_version']
            stuff['usage'] = rd['usage_metadata']
            for k, v in choice['parts'][0].items():
                if k == 'text':
                    stuff['message'] += v
                else:
                    print(f"in gemini, unk resp k,v: = {k}, {v} ...")
            stuff['finish-reason'] = rd['candidates'][0]['finish_reason']
            llm_client = 'gemini'
        else:
            raise NotImplementedError(f"parse_resp for client: {self.client!r} not implemented...")
        # logic is the same across clients
        stuff['llm_client'] = llm_client
        return stuff


class Prompt(Template):
    def __init__(self, template):
        super().__init__(template)

# given in https://arxiv.org/pdf/2406.06519 fig 1
# NOTE: using "gemini-1.0-pro-002" w/this prompt gets it  to using the bing
#  style prompt output from:
#  https://github.com/castorini/umbrela/tree/main/src/umbrela/prompts
_umb_promt = """
Given a query and a passage, you must provide a score on an
integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but
does not answer it,
2 = represents that the passage has some answer for the query,
but the answer may be a bit unclear, or hidden amongst extraneous
information and
3 = represents that the passage is dedicated to the query and
contains the exact answer.
Important Instruction: Assign category 1 if the passage is
somewhat related to the topic but not completely, category 2 if
passage presents something very important related to the entire
topic but also has some extra information and category 3 if the
passage only and entirely refers to the topic. If none of the
above satisfies give it category 0.
Query: $query
Passage: $passage
Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query
(M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each,
and decide on a final score (O). Final score must be an integer
value only.
Do not provide any code in result. Provide each score in the
format of:

##final score: score without providing any reasoning.
"""

_llm_rel_assess = """
You are a search quality rater evaluating the relevance
of web pages. Given a query and a web page, you must
provide a score on an integer scale of 0 to 2 with the
following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain
other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the
topic. If you would use any of the information contained
in the web page in such a report, mark it 1. If the web page
is primarily about the topic, or contains vital information
about the topic, mark it 2. Otherwise, mark it 0.

Query
A person has typed
$query
into a search engine.

They were looking for:
$narrative

Result
Consider the following web page.
—BEGIN CONTENT—
$passage
—END CONTENT—
Instructions
Split this problem into steps:

Measure how well the content matches a likely intent of
the query (M).

Measure how trustworthy the web page is (T).

Consider the aspects above and the relative importance
of each, and decide on a final score (O).

Produce a JSON array of scores without providing any
reasoning. Example: [{"M": 2, "T": 1, "O": 1}, {"M":
1 . . .
Results
[{
"""

# Note the narrative for the second prompt is unlikely to be provided
# given that we do not have a browser worflow interface setup for this part-yet.

# TODO: make the narrative optional
# TODO: implement the narrative input workflow


if __name__ == "__main__":
    ## NOTE: intended for lightweight spot checking during dev work only...
    prompt = Prompt(_umb_promt)
    llm_rel = LLMRelAssessor(prompt=prompt)
    # NOTE: if you switch passage with, say 'ok?' the returned score flips 0 -> 3...
    tp = {'query': 'Say this is a test', 'passage': 'say this is a test'}
    resp = llm_rel.generate_rel(template_params=tp)
    # NOTE: not sure if other llm clients have the `to_dict` on their responses...
    rd = resp.to_dict()
    if len(rd['choices']) > 1:
        print("multiple choices in response, taking the first...")
    choice = rd['choices'][0]
    stuff = {
        'llm_client': 'openai',
        'msg-id': rd['id'],
        'model-id': rd['model'],
        'finish-reason': choice['finish_reason'],
        'message': choice['message']['content'],
        'usage': rd['usage']
    }
    print(stuff)