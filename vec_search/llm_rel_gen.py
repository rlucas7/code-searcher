"""
This module houses the framework for generating LLM relevances for search queries.
"""

import os
import json

from abc import ABC
from collections import defaultdict
from pathlib import Path
from string import Template
from time import sleep
from typing import Union

from pandas import DataFrame, concat

# llm clients
import vertexai

from openai import OpenAI
from vertexai.batch_prediction import BatchPredictionJob
from google.cloud import storage


class LLMRelAssessorBase(ABC):
    def __init__(
        self,
        df: DataFrame,
        output_filename: str,
        prompt: Union[str, Template] = "",
        model_name: str = "",
        shot_count: int = 0,
    ):
        self.shot_count = shot_count
        self.model_name = model_name
        self.prompt = prompt if Template else Template(prompt)
        self.df = df
        self.output_filename = output_filename

    def _create_client(self):
        raise NotImplementedError(
            "error: you must implement this method in a child class ..."
        )


class LLMRelAssessor(LLMRelAssessorBase):
    def __init__(
        self,
        # qrel,
        df: DataFrame,
        output_filename: str,
        prompt: Union[str, Template] = "",
        model_name: str = "openai",
        shot_count: int = 0,
    ):
        super().__init__(
            df=df,
            output_filename=output_filename,
            shot_count=shot_count,
            prompt=prompt,
            model_name=model_name,
        )
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
            PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
            LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            # NOTE: (i) model = "gemini-1.0-pro" # does not comport well with prompt
            # (ii) api limits are strict so we need batch which requires gcs store input
            model = "gemini-1.5-pro-001"
            self.client = BatchPredictionJob
            self.model_name = model
        elif model_name == "aws":
            # NOTE: aws code is in bedrock_batch.py
            self.model_name = "us.amazon.nova-lite-v1:0"
            self.client = None
        else:
            raise ValueError(f"model {self.model_name} is not currently supported ...")

    def generate_rel(self, parse: bool = True):
        if isinstance(self.client, OpenAI):
            ## generate the llm relevance determinations
            llm_gen_data = defaultdict(list)
            for index, row in self.df.iterrows():
                print(f"processing index: {index} ...")
                # access of all entries follows via column name as key
                query = row['query']
                passage = row['doc'] + "\n\n\n" + row['code']
                tp = {'query': query, 'passage': passage}

                content = self.prompt.safe_substitute(tp)
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    model=self.model_name,
                )
                resp = self.parse_resp(response) if parse else response
                for key, value in resp.items():
                    llm_gen_data[key].append(value)
            llm_gen_df = DataFrame(llm_gen_data)
            c_df = concat([llm_gen_df, self.df], axis=1)
            # if the respon score is not 0/1 then convert it
            c_df["llm_rel_score"] = (
                c_df["message"]
                .str.split(":")
                .apply(lambda x: max(0, min(int(x[1].strip()), 1)))
            )
            # write results to local filesystem
            c_df.to_csv(self.output_filename)
        elif self.model_name == "gemini-1.5-pro-001":
            input_bucket_name = "batch-llm-relevance-inputs"
            output_bucket_name = "gemini-completions-batch"
            storage_client = storage.Client()
            # format df and write to `input_bucket`
            with open("./examples.jsonl", "w") as f:
                for index, row in self.df.iterrows():
                    query = row['query']
                    passage = row['doc'] + "\n\n\n" + row['code']
                    tp = {'query': query, 'passage': passage}
                    content = self.prompt.safe_substitute(tp)
                    r = {
                        "request": {
                            "contents": [
                                {"role": "user", "parts": [{"text": content}]}
                            ],
                            "generationConfig": {"temperature": 0.4},
                        }
                    }
                    # request as a string and write to tempfile.
                    rs = json.dumps(r)
                    f.write(rs + "\n")
            # now upload the file to gcs
            blob = storage_client.bucket(input_bucket_name).blob("examples.jsonl")
            blob.upload_from_filename("./examples.jsonl")
            print("upload to gcs finished...")
            sleep(10)
            batch_prediction_job = BatchPredictionJob.submit(
                source_model=self.model_name,
                # for now make these const
                input_dataset=f"gs://{input_bucket_name}/examples.jsonl",
                output_uri_prefix=f"gs://{output_bucket_name}",
            )
            # Check job status
            print(f"Job resource name: {batch_prediction_job.resource_name}")
            print(
                f"Model resource name with the job: {batch_prediction_job.model_name}"
            )
            print(f"Job state: {batch_prediction_job.state.name}")

            # Refresh the job until complete
            while not batch_prediction_job.has_ended:
                sleep(30)
                batch_prediction_job.refresh()
                # this seems preferrable to the given example if/elif beneath
                print(f"Job state: {batch_prediction_job.state.name}")
                # Check if the job succeeds
                if batch_prediction_job.has_succeeded:
                    print("Job succeeded!")
                elif batch_prediction_job.error != "":
                    print(f"Job failed: {batch_prediction_job.error} ...")
            # Give the location of the output in gcs
            print(f"Job output location: {batch_prediction_job.output_location}")
            # now given the output location we need to parse and handle...
            bucket = storage_client.bucket(output_bucket_name)
            # the convention seems to be this...
            blob_name = (
                batch_prediction_job.output_location.split("/")[-1]
                + "/predictions.jsonl"
            )
            blob = bucket.blob(blob_name)
            data = []
            with blob.open("r") as f:
                for line in f:
                    data.append(json.loads(line))
            # construct parsed data
            p_data = []
            for entry in data:
                p_data.append(self.parse_resp(entry["response"]))
            df = DataFrame(p_data)
            c_df = concat([df, self.df], axis=1)
            c_df["llm_rel_score"] = (
                df["message"]
                .str.split(":")
                .apply(lambda x: max(0, min(int(x[1].strip()), 1)))
            )
            c_df.to_csv(self.output_filename)
        elif self.model_name == "us.amazon.nova-lite-v1:0":
            from .bedrock_batch import bb
            print("aws bedrock batch workflow starting ...")
            bb(df=self.df, prompt=self.prompt, output_filename=self.output_filename)
        else:
            raise NotImplementedError(
                f"generate_rel for client: {self.client!r} not implemented..."
            )

    def parse_resp(self, resp) -> dict[str, str]:
        # openai docs on response object
        # https://platform.openai.com/docs/api-reference/chat/object
        # NOTE: maybe type this as a 'ParsedResponse'...
        stuff = {
            "llm_client": "",
            "msg-id": "",
            "model-id": "",
            "finish-reason": "",
            "message": "",
            "usage": "",
        }
        # both OpenAI and vertexai.GenerativeModel have a .to_dict() method
        if isinstance(self.client, OpenAI):
            rd = resp.to_dict()
            if len(rd["choices"]) > 1:
                print("multiple choices in response, taking the first...")
            choice = rd["choices"][0]
            # stuff ...
            stuff["msg-id"] = rd["id"]
            stuff["model-id"] = rd["model"]
            stuff["usage"] = rd["usage"]
            stuff["message"] = choice["message"]["content"]
            stuff["finish-reason"] = choice["finish_reason"]
            # ...
            llm_client = "openai"
        elif self.model_name == "gemini-1.5-pro-001":
            # with gemini batch replies-we've already converted to dict
            # NOTE: batch keynames are camelcased whereas non-batch are snake-arg!
            rd = resp
            if len(rd["candidates"]) > 1:
                print("multiple candidates in response, taking the first...")
            choice = rd["candidates"][0]["content"]
            stuff["msg-id"] = None
            stuff["model-id"] = rd.get("modelVersion", self.model_name)
            stuff["usage"] = rd.get("usageMetadata", "")
            for k, v in choice["parts"][0].items():
                if k == "text":
                    stuff["message"] += v
                else:
                    print(f"in gemini, unk resp k,v: = {k}, {v} ...")
            stuff["finish-reason"] = rd["candidates"][0]["finishReason"]
            llm_client = "gemini"
        else:
            raise NotImplementedError(
                f"parse_resp for client: {self.client!r} not implemented..."
            )
        # logic is the same across clients
        stuff["llm_client"] = llm_client
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
