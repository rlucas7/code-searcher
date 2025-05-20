""" This module cli script contains code to tune a prompt
using the cosQA data.

NOTE: The tuning takes several hours to execute a single epoch on
the dev data (604 relevance records).

Be sure to set the openAI api key env var before executing this
cli script.
"""

import logging
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import deque
from json import loads

import textgrad as tg

from requests import get as RequestsGet

logger = logging.getLogger()

# logger to stdout (console) and to file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
file_handler = logging.FileHandler("prompt_tuning.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(lineno)d::(message)s'))
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


    ### TODO: modify this starting prompt to take the prompt from the project, link is here
    ### https://github.com/rlucas7/code-searcher/blob/08ba27f859c0450117e78e74edb2192454ca23fb/vec_search/llm_rel_gen.py#L275

    system_prompt = tg.Variable("You are a code Search results analyst. I will give you a query and a search result with a [0, 1] for whether the result is relevant for the query, 0 indicates an irrelevant result while 1 indicates a relevant result. Your output should exactly match the given values. No extra explanation, just the relevance category.", requires_grad=True, role_description="system prompt to the language model")
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
            logger.info("len(system_prompt.value):", len(system_prompt.value))

    # now evaluate on the dev data
    up_prompt_relevances = []
    og_prompt_relevances = []
    og_system_prompt = tg.Variable("You are a code Search results analyst. I will give you a query and a search result with a [0, 1] for whether the result is relevant for the query, 0 indicates an irrelevant result while 1 indicates a relevant result. Your output should exactly match the given values. No extra explanation, just the relevance category.", requires_grad=False, role_description="original system prompt to the language model")
    og_llm = tg.get_engine(engine_name=args.llm_engine_name)
    og_model = tg.BlackboxLLM(og_llm, system_prompt=og_system_prompt)

    dev_loader = zip([dev_contents[idx]["doc"] + " & " + dev_contents[idx]["code"] for idx in range(N_dev)], [dev_contents[idx]["label"] for idx in range(N_dev)])
    for i in range(N_dev):
        logger.info(f"evaluating {i}th record with original and optimized prompts")
        query_n_result_x, relevance_z = next(train_loader)
        x = tg.Variable(query_n_result_x, requires_grad=False, role_description="code search query & result")
        z = tg.Variable(relevance_z, requires_grad=False, role_description="ground truth relevance determination of the result for the query")
        resps = model(x)
        og_resps = og_model(x)
        up_prompt_relevances.append(resps.value)
        og_prompt_relevances.append(og_resps.value)

    assert len(up_prompt_relevances) == len(og_prompt_relevances), f"Error two output relevance lists are not equal length"


    ### Ok if that indeed 'optimized' the system prompt in some sense, then we expect
    ### to see an improvement in metrics on a hold out set of relevances w.r.t. the
    ### optimized prompt relative to the base prompt.



    ### Compare the relevance determinated based on the two prompts, default and optimized
    ### against the actual relevances. Which of the two prompts results in better metrics?

    actual_relevances = [str(dev_contents[idx]["label"]) for idx in range(N_dev)]
    # NOTE: In testing the optimized prompt seems maintina the 0/1 relevance at the beginning
    # of the prompt. If that changes during the optimization step this part will break.
    up_relevances = [up_prompt_relevances[idx][0:1] for idx in range(N_dev)]


    # first compare actuals with originals

    actual_vs_naive = 0
    actual_vs_naive_pos = 0
    for actual, naive in zip(actual_relevances, og_prompt_relevances):
        val = 1 if actual == naive else 0
        val_pos = 1 if actual == naive and actual == '1' else 0
        actual_vs_naive += val
        actual_vs_naive_pos += val_pos

    actual_vs_opt = 0
    actual_vs_opt_pos = 0
    for actual, opt in zip(actual_relevances, up_relevances):
        val = 1 if actual == opt else 0
        val_pos = 1 if actual == opt and actual == '1' else 0
        actual_vs_opt += val
        actual_vs_opt_pos += val_pos


    logger.info(f"""accuracy of naive prompt: {actual_vs_naive/N_dev} \n
    accuracy of optimized prompt: {actual_vs_opt/N_dev} \n
    accuracy of naive prompt negative proportion: {(actual_vs_naive - actual_vs_naive_pos)/N_dev} \n
    accuracy of optimized prompt negative proportion: {(actual_vs_opt - actual_vs_opt_pos)/N_dev} \n
    accuracy of naive prompt positive proportion: {actual_vs_naive_pos/N_dev} \n
    accuracy of optimized prompt positive proportion: {actual_vs_opt_pos/N_dev} \n
    """)