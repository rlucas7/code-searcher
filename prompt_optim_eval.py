""" evaluates two prompts on a given dataset.
The default is to use:
https://raw.githubusercontent.com/rlucas7/code-searcher/refs/heads/main/cosqa-train-1.json
which is the first 20% of the cosQA data from:
https://raw.githubusercontent.com/Jun-jie-Huang/CoCLR/refs/heads/main/data/qa/cosqa-train.json
if you want to run this on a subsequent portion invoke with:

python prompt_optim_eval.py --eval_data_url https://raw.githubusercontent.com/rlucas7/code-searcher/refs/heads/main/cosqa-train-2.json
for the second 20%, changing the integer just prior to the .json extension is all that is needed.
"""

import logging
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from json import loads

import textgrad as tg

from requests import get as RequestsGet

if not os.environ.get("OPENAI_API_KEY"):
    e = EnvironmentError(f"The OPENAI_API_KEY env var is not set. Please set the api key and the retry.")
    raise e

parser = ArgumentParser(
    prog='Prompt eval script. Given a base prompt and an optimized version, e.g. via the `prompt_tune.py` module. ' \
    'Feed both through an LLM and evaluate on given evaluation data.',
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="A script to evaluate two prompts using an eval dataset. " \
    "By default we use cosQA data." \
    "Small modifications to the script would be needed to use different datasets."
    )

hold_out_data_url = "https://raw.githubusercontent.com/rlucas7/code-searcher/refs/heads/main/cosqa-train-1.json"
parser.add_argument("--eval_data_url", type=str, help="The url to access the hold out data in json form.", default=hold_out_data_url)
parser.add_argument("--llm_engine_name", type=str, help="The name of the llm engine used by textgrad for prompt tuning.", default="gpt-4o-mini")
parser.add_argument("--prompt_input_files", nargs=2, type=str, help="The filenames which will contain two prompts to be compared. " \
    "The convention assumed in this script is that the first file contains the optimized prompt and the second is the pre-optimized prompt.",
    default=["optimized_prompt.txt", "original_prompt.txt"])
parser.add_argument("--verbose_eval_logging", action="store_true", help="Setting this flag will cause all LLM call prompts from TextGrad to be emitted to the logs. " \
"Turned off by default because it makes grepping logs difficult and bloats the log files excessively. The output can be helpful for debugging issues on small sets of test cases.")
# handles stdout only ATM


if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose_eval_logging:
        # taken from https://stackoverflow.com/a/75373988/3164100
        # The net effect here is to not raise a key error if the log record does not contain the key 'text' and to emit the value
        # associated with the key if the log record does contain this value. The reason we need to do it this way is because this
        # is how textgrad has (currently) decided to implement the log record for the forward calls
        # https://github.com/zou-group/textgrad/blob/bf5b0c5df0d029293dcc355a389a55908c61a9c6/textgrad/autograd/llm_ops.py#L69
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s: %(text)s')
        class NewLogger(logging.Logger):
            def makeRecord(self, *args, **kwargs):
                rv = super(NewLogger, self).makeRecord(*args, **kwargs)
                rv.__dict__["text"] = rv.__dict__.get("text", "<None>")
                return rv
        logger = NewLogger(__name__)
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)

    # load eval data
    logging.info(f"Getting data from {args.eval_data_url}")
    dev_resp = RequestsGet(args.eval_data_url)
    assert dev_resp.status_code==200, f"Got an error code from Github: {dev_resp.status_code}, maybe .. try again?"
    eval_contents = loads(dev_resp.content)
    N_dev = len(eval_contents)
    logger.info(f"downloaded eval data with lengths: {N_dev}")
    up_prompt_relevances = []
    og_prompt_relevances = []
    # read in the two prompts to be compared from files
    opt_p_file, orig_p_file = args.prompt_input_files
    logger.info(f"reading prompts to compare from files: {opt_p_file} and {orig_p_file}")
    with open(opt_p_file, "r") as fh1:
        opt_p_text = fh1.read()
    with open(orig_p_file, "r") as fh2:
        orig_p_text = fh2.read()
    og_system_prompt = orig_p_text
    system_prompt = opt_p_text
    logger.info("--------------optimized prompt--------------\n\n")
    logger.info(opt_p_text)
    logger.info("--------------original prompt--------------\n\n")
    logger.info(orig_p_text)


    # now finished setting things up...
    og_system_prompt = tg.Variable(orig_p_text, requires_grad=False, role_description="original system prompt to the language model")
    og_llm = tg.get_engine(engine_name=args.llm_engine_name)
    og_model = tg.BlackboxLLM(og_llm, system_prompt=og_system_prompt)
    llm = tg.get_engine(engine_name=args.llm_engine_name)

    model = tg.BlackboxLLM(llm, system_prompt=system_prompt)
    dev_loader = zip([eval_contents[idx]["doc"] + " & " + eval_contents[idx]["code"] for idx in range(N_dev)], [eval_contents[idx]["label"] for idx in range(N_dev)])
    # do the eval calculations...
    th_map = {1: "st", 2: "nd", 3: "rd"} # use th_map.get(int, "th") in logging... still has some errors but close enough for now..
    for i, (query_n_result_x, relevance_z) in enumerate(dev_loader, start=1):
        logger.info(f"evaluating {i}{th_map.get(i%10, 'th')} record with original and optimized prompts")
        logger.debug(f"query_n_result for record {i} is: {query_n_result_x}")
        x = tg.Variable(query_n_result_x, requires_grad=False, role_description="code search query & result")
        z = tg.Variable(relevance_z, requires_grad=False, role_description="ground truth relevance determination of the result for the query")
        resps = model(x)
        og_resps = og_model(x)
        logger.debug(f"resp.value is {resps.value} and og_resps.value is: {og_resps.value} and actual is: {z.value}")
        up_prompt_relevances.append(resps.value)
        og_prompt_relevances.append(og_resps.value)
    if len(up_prompt_relevances) != len(og_prompt_relevances):
        logging.info(f"Error two output relevance lists are of lengths {len(up_prompt_relevances)} and {len(og_prompt_relevances)}, additional elements in the longer list will be skipped")

    ### Compare the relevance determinated based on the two prompts, default and optimized
    ### against the actual relevances. Which of the two prompts results in better metrics?
    actual_relevances = [str(eval_contents[idx]["label"]) for idx in range(N_dev)]

    # NOTE: In testing the optimized prompt seems maintian the 0/1 relevance at the end, 'Final Score: 0/1' is what I see.
    # of the prompt. If that changes during the optimization step this part will break.
    # TODO: investigate structured output call here to mitigate
    up_relevances = [up_prompt_relevances[idx][-1] for idx in range(N_dev)]
    ogp_relevances = [og_prompt_relevances[idx][-1:] for idx in range(N_dev)]
    # first compare actuals with originals

    actual_vs_naive = 0
    actual_vs_naive_pos = 0
    for i, (actual, naive) in enumerate(zip(actual_relevances, ogp_relevances), start=1):
        val = 1 if actual == naive else 0
        val_pos = 1 if actual == naive and actual == '1' else 0
        actual_vs_naive += val
        actual_vs_naive_pos += val_pos

    actual_vs_opt = 0
    actual_vs_opt_pos = 0
    for i, (actual, opt) in enumerate(zip(actual_relevances, up_relevances), start=1):
        val = 1 if actual == opt else 0
        val_pos = 1 if actual == opt and actual == '1' else 0
        actual_vs_opt += val
        actual_vs_opt_pos += val_pos


    logger.info(f"accuracy of naive prompt: {actual_vs_naive/N_dev}")
    logger.info(f"accuracy of optimized prompt: {actual_vs_opt/N_dev}")
    logger.info(f"accuracy of naive prompt negative proportion: {(actual_vs_naive - actual_vs_naive_pos)/N_dev}")
    logger.info(f"accuracy of optimized prompt negative proportion: {(actual_vs_opt - actual_vs_opt_pos)/N_dev}")
    logger.info(f"accuracy of naive prompt positive proportion: {actual_vs_naive_pos/N_dev}")
    logger.info(f"accuracy of optimized prompt positive proportion: {actual_vs_opt_pos/N_dev}")