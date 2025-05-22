""" This module cli script contains code to tune a prompt
using the cosQA data.

NOTE: The tuning takes several hours to execute a single epoch on
the dev data (604 relevance records).

Be sure to set the openAI api key env var before executing this
cli script. E.g. run
`export OPENAI_API_KEY="<your-api-key>"`
before running.
"""

import logging
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import deque
from json import loads

import textgrad as tg

from requests import get as RequestsGet

assert os.environ["OPENAI_API_KEY"], f"set your open ai key via: export `OPENAI_API_KEY=\"<your-api-key>\"` and rerun the cli script."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# logger to stdout (console) and to file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
file_handler = logging.FileHandler("prompt_tuning.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# NOTE: this part is a bit confusing when you read it because the cosQA `dev` data are used for training
# and the cosQA `train` data are used as hold out for metric calculations. This is for reasons of scale.
# The I/O for the 604 records in the dev data is enough and already takes several hours for 1 epoch
# of tuning the prompt. The 20K records in the `train` data would take many days/weeks to perform tuning.

train_data_url = "https://raw.githubusercontent.com/Jun-jie-Huang/CoCLR/refs/heads/main/data/qa/cosqa-dev.json"
hold_out_data_url = "https://raw.githubusercontent.com/Jun-jie-Huang/CoCLR/refs/heads/main/data/qa/cosqa-train.json"

parser = ArgumentParser(
    prog='Prompt tuning script',
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="A script to tune a prompt using a training and dev dataset. " \
    "By default we use cosQA data." \
    "Small modifications to the script would be needed to use different datasets."
    )

parser.add_argument("--train_data_url", type=str, help="The url to access the training data in json form.", default=train_data_url)
parser.add_argument("--hold_out_data_url", type=str, help="The url to access the hold out data in json form.", default=hold_out_data_url)
parser.add_argument("--llm_engine_name", type=str, help="The name of the llm engine used by textgrad for prompt tuning.", default="gpt-4o-mini")
parser.add_argument("--batch_size", type=int, help="The size of each batch for the prompt tuning.", default=4)
parser.add_argument("--prompt_output_file", type=str, help="The filename for the output file which will contain the optimized prompt.", default="optimized_prompt.txt")

if __name__ == "__main__":
    args = parser.parse_args()
    train_resp = RequestsGet(args.train_data_url)
    assert train_resp.status_code==200, f"Got an error code from Github: {train_resp.status_code}, maybe .. try again?"

    train_contents = loads(train_resp.content)
    N_train = len(train_contents)
    prop_ones = sum([train_contents[i]['label'] for i in range(N_train)]) / N_train
    logger.info(f"Proportion of relevant results in: \n {train_data_url[-15:-1]} \n is {prop_ones} ")

    dev_resp = RequestsGet(args.hold_out_data_url)
    assert dev_resp.status_code==200, f"Got an error code from Github: {train_resp.status_code}, maybe .. try again?"
    dev_contents = loads(dev_resp.content)
    N_dev = len(dev_contents)
    logger.info(f"downloaded dev and train data for prompt tuning with lengths: {N_dev} and {N_train} respectively")
    ## NOTE(for the reader): dev_contents[i]["doc"] is the query (it's a confusing name)
    ## and dev_contents[i]["code"] is the python function
    ## and dev_contents[i]["label"] is the 0/1 relevance label,
    ## 1 encodes relevant result for query and 0 encodes not relevant

    ## end of load in code

    tg.set_backward_engine(args.llm_engine_name)

    fmt_string = """
    LLM Prompt: {system_prompt}
    Query & Result: {query_n_result}
    Prediction: {pred}
    Ground Truth: {actual}
    Evaluation: {eval}
    """

    loss_system_prompt = tg.Variable("""Your job is to provide feedback to an LLM classifier.
    You will get the search query, a search result, the LLM generated relevance, as well as the ground truth relevance.
    The LLM output should EXACTLY match the ground truth target, and the eval Evaluation should be True.
    You must provide concise feedback to correct the response.
    """, role_description = "System prompt to provide feedback", requires_grad=False
    )

    fields = {"system_prompt": None, "query_n_result": None,"pred": None, "actual": None, "eval": None}

    formatted_llm_call = tg.autograd.FormattedLLMCall(engine=tg.get_engine(engine_name=args.llm_engine_name), format_string=fmt_string, fields = fields, system_prompt=loss_system_prompt)


    ### https://github.com/rlucas7/code-searcher/blob/08ba27f859c0450117e78e74edb2192454ca23fb/vec_search/llm_rel_gen.py#L275
    sys_prompt_text = """Given a query and a passage, you must provide a score on an
integer scale of 0 to 1 with the following meanings:
0 = represent that the passage has nothing to do with the query,
0 = represents that the passage seems related to the query but
does not answer it,
1 = represents that the passage has some answer for the query,
but the answer may be a bit unclear, or hidden amongst extraneous
information and
1 = represents that the passage is dedicated to the query and
contains the exact answer.
Important Instruction: Assign category 0 if the passage is
somewhat related to the topic but not completely, category 1 if
passage presents something very important related to the entire
topic but also has some extra information and category 1 if the
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
    system_prompt = tg.Variable(sys_prompt_text, requires_grad=True, role_description="system prompt to the language model")
    llm = tg.get_engine(engine_name=args.llm_engine_name)
    model = tg.BlackboxLLM(llm, system_prompt=system_prompt)
    optimizer = tg.TextualGradientDescent(engine=llm, parameters=[system_prompt])


    def loss_fn(system_prompt, query, pred, target, _eval):
        return formatted_llm_call(inputs={"system_prompt": system_prompt,
            "query_n_result": query,
            "pred": pred,
            "actual": target,
            "eval": _eval
            })


    train_loader = zip([train_contents[idx]["doc"] + " & " + train_contents[idx]["code"] for idx in range(N_train)], [train_contents[idx]["label"] for idx in range(N_train)])
    # `maxlen=batch_size` makes it auto-drop the older items
    losses = deque(maxlen=args.batch_size)

    for i in range(N_train):
        query_n_result_x, relevance_z = next(train_loader)
        query_n_result = tg.Variable(query_n_result_x, requires_grad=False, role_description="code search query & result")
        relevance = tg.Variable(relevance_z, requires_grad=False, role_description="ground truth relevance determination of the result for the query")
        resps = model(query_n_result)
        if resps.value.lower() == relevance.value.lower():
            eval_str = "Correct as prediction exactly matches target"
        else:
            eval_str = f"Incorrect as prediction doesn't exactly match target. LLM response: {resps.value} vs Actual value: {relevance.value}."
        Eval = tg.Variable(eval_str, requires_grad=False, role_description="Evaluation")
        losses.append(loss_fn(system_prompt=system_prompt, query=query_n_result, pred=resps, target=relevance, _eval=Eval))
        if i > 0 and i % args.batch_size == 0:
            optimizer.zero_grad()
            logger.info(f"optimizing w/i={i}")
            logger.info(losses)
            ttl_loss = tg.sum(losses)
            ttl_loss.backward()
            optimizer.step()
            logger.info(f"length of system_prompt: {len(system_prompt.value)}")
            if i % (3 * args.batch_size) == 0:
                logger.info(f"iteration i = {i} has system_prompt:\n {system_prompt.value}")

    # now write the optimized prompt the filesystem for reuse in evaluations
    with open(args.prompt_output_file, "w") as fh:
        fh.writelines(system_prompt.value)
    logger.info(f"optimized prompt written to output file: {args.prompt_output_file} successfully")
    logger.info("program complete-all done")
