""" This module contains the code for the bedrock batch request workflow
"""
import os

from enum import Enum
from string import Template
from typing import Optional, Any
from time import sleep

import boto3
import uuid
import json

from pandas import DataFrame, notna, concat
from pydantic import BaseModel, Field, ConfigDict


# some helper codes
def gen_record_id(index: int, prefix: str = "REC") -> str:
    """Generate an 11 character alphanumeric record ID."""
    return f"{prefix}{str(index).zfill(8)}"

def parse_resp(x: list[str]) -> int:
    """

    Args:
        x (list[str]): A string with the llm response from the umbrella prompt.

    Raises:
        ValueError: if x is not a string.

    Returns:
        int: A binary 0/1 value for whether the model determined the relevance (1) or not (0)
    """
    try:
        rel = max(0, min(int(x[1].strip()), 1))
    except ValueError as e:
        print(f"ValueError: {e}...")
        rel = 0
    return rel

class ModelType(Enum):
    TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
    NOVA_LITE = "us.amazon.nova-lite-v1:0"


class BaseGenerationConfig(BaseModel):
    # settings matche gemini settings we use
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 512
    stop_sequences: Optional[list[str]] = Field(default_factory=list)
    top_k: Optional[int] = 40 # gemini's default
    system: Optional[str] = None
    model_config = ConfigDict(
        populate_by_name=True
    )

class NovaPrompt(BaseModel):
    inputText: list[dict[str, Any]]
    textGenerationConfig: dict

    def model_dump(self, *args, **kwargs) -> dict:
        # the dict formats need to follow this:
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        return {
            "messages": self.inputText,
            "inferenceConfig": {
                 "maxTokens": 512,
                 "stopSequences": [],
                 "temperature": 0.4,
                 "topP": 0.99,
            },
        }

def dataframe_to_jsonl(
    df: DataFrame,
    model_type: ModelType,
    output_file: str,
    prompt: Template,
    record_id_column: Optional[str] = None,
    base_config: Optional[BaseGenerationConfig] = None
) -> None:
    """
    Convert a DataFrame to a JSONL file for batch inference.

    Args:
        df: Input DataFrame containing text and optional configuration columns
        model_type: Type of model to generate prompts for
        output_file: Path to save the JSONL file
        prompt:  The prompt containing input text
        record_id_column: Optional column name containing record IDs
        base_config: Default configuration to use for missing values
    """
    if base_config is None:
        base_config = BaseGenerationConfig()

    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            # Get or generate record ID
            record_id = str(row[record_id_column]) if record_id_column and record_id_column in row else gen_record_id(idx)
            # Create config from row data, falling back to base_config for missing values
            config_dict = base_config.model_dump()
            for field in config_dict.keys():
                if field in row and notna(row[field]):
                    config_dict[field] = row[field]

            row_config = BaseGenerationConfig(**config_dict)
            # Create batch inference record
            # similar logic across: concat docstring and code
            passage = row['doc'] + "\n\n\n" + row['code']
            tp = {'query': row['query'], 'passage': passage}
            content = prompt.safe_substitute(tp)

            body = [{"role": 'user',
                     "content": [{'text': content}]
                    }]
            record = {
                "recordId": record_id,
                "modelInput": get_request_body(
                    text=body,
                    model_type=model_type,
                    config=row_config
                )
            }
            f.write(json.dumps(record) + '\n')


def get_request_body(text: list[dict[str, Any]], model_type: ModelType, config: Optional[BaseGenerationConfig] = None) -> NovaPrompt:
    if config is None:
        config = BaseGenerationConfig()

    if model_type == ModelType.NOVA_LITE:
        return NovaPrompt(
            inputText=text,
            textGenerationConfig={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stop_sequences": config.stop_sequences
            }
        ).model_dump()
    raise ValueError(f"Unknown model type: {model_type}")


def bb(df, prompt, output_filename):
    """executes a batch request against aws bedrock service

    Args:
        df: A pandas dataframe with the relevant data from human annotations
        prompt: A Template string with relevance annotations instruction from humans
    """
    config = BaseGenerationConfig(
        temperature=0.4,
        max_tokens=512,
        stop_sequences=[],
        system="You are a helpful assistant. Please use the context to answer the question that follows."
    )

    ACCESS_KEY=os.environ.get("ACCESS_KEY")
    SECRET_KEY=os.environ.get("SECRET_KEY")
    AWS_REGION=os.environ.get("AWS_REGION")
    AWS_ACCT_ID = os.environ.get("AWS_ACCT_ID")
    BUCKET_NAME = "annotate-io-batch-llm-data"
    NOVA = "us.amazon.nova-lite-v1:0"
    # this is iam-role, we need a role
    # so we can delegate to the batch service
    arn = f"arn:aws:iam::{AWS_ACCT_ID}:role/bedrock-batch-role"
    model_id = ModelType(NOVA)
    bucket_prefix = f"batch_requests_{model_id}-lrr.jsonl"
    # TODO: cleanup the format for the input to the batch job here
    dataframe_to_jsonl(
        df=df,
        model_type=model_id,
        output_file='./examples.jsonl', # same filename
        prompt=prompt,
        base_config=config
    )
    print(f"Uploading to s3://{BUCKET_NAME}/{bucket_prefix}")
    s3 = boto3.client('s3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=AWS_REGION
    )
    s3.upload_file(Filename='./examples.jsonl', Bucket=BUCKET_NAME, Key=bucket_prefix)
    print(f"Uploading to s3 complete")
    # added from here down
    print("invoking bedrock for relevances")
    inputDataConfig=({
        "s3InputDataConfig": {
            "s3Uri": f"s3://{BUCKET_NAME}/{bucket_prefix}"
        }
    })

    outputDataConfig=({
        "s3OutputDataConfig": {
            "s3Uri": f"s3://{BUCKET_NAME}/"
        }
    })

    bedrock = boto3.client('bedrock',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=AWS_REGION
    )
    print("invoking batch job")
    # get first part of the uuid
    a_uuid = hex(uuid.uuid4().fields[0])[2:]
    response = bedrock.create_model_invocation_job(
        roleArn=arn,
        modelId=model_id.NOVA_LITE.value,
        jobName=f"nova-eval-{a_uuid}",
        inputDataConfig=inputDataConfig,
        outputDataConfig=outputDataConfig
    )
    print(response)
    jobArn = response.get('jobArn')
    poll_cnt, max_poll_cnt = 1, 20

    while poll_cnt < max_poll_cnt:
        sleep(30)
        jobInfo = bedrock.get_model_invocation_job(jobIdentifier=jobArn)
        if jobInfo['status'] == "Completed":
            print(f"job: {jobArn} finished")
            break
        else:
            print(f"job: {jobArn} is in state: {jobInfo['status']}")
            poll_cnt += 1
            print("---")
    else:
        print(f"max polling iteration: {max_poll_cnt} reached ...")

    if jobInfo['status'] == "Completed":
        # the jobArn characters are the same as the results folder in bucket
        jobDir = jobArn.split('/')[-1]
        batch_out = f"{jobDir}/{bucket_prefix}.out"
        s3.download_file(BUCKET_NAME, batch_out, f"bedrock_batch_output_{jobDir}.jsonl")
        print("output file downloaded")
        # TODO: here parse the output file into the desired format...
        data = []
        with open(f"bedrock_batch_output_{jobDir}.jsonl", 'r') as f:
            for line in f:
                rd = json.loads(line)
                data.append(rd['modelOutput']['output']['message']['content'][0]['text'])
        # construct parsed data
        p_df = DataFrame({'message': data})
        c_df = concat([df, p_df], axis=1)
        c_df['llm_rel_score'] = c_df['message'].str.split(':').apply(parse_resp)
        c_df.to_csv(output_filename)