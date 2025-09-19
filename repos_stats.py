"""
This module is intended to gather and analyze statistics from code repositories.
The code assumes the existence of repositories in JSONL format, where each line is a JSON object with the following fields:
- func_name: The name of the function.
- original_string: The original source code of the function as a string.
- docstring_tokens: A list of tokens from the function's docstring.
- code_tokens: A list of tokens from the function's code (excluding the docstring).
"""

import argparse

from collections import Counter

import jsonlines



repos_stats_parser = argparse.ArgumentParser(description="A simple script to calculate stats on indexed repos.")

repos_stats_parser.add_argument("--inputfile", type=str, help="The file name where the indexed repo stored.")


if __name__ == "__main__":
    args = repos_stats_parser.parse_args()
    # NOTE: these names are referenced by index later so if you add new ones add at end of list
    # and for each new statistic calculated, add a new entry to the stats dict below inside the
    # context manager which loops over the indexed functions in the repo
    stats_names = ["total_funcs", "total_lines_of_code", "total_docstring_tokens", "total_code_tokens"]
    stats = {name: 0 for name in stats_names}
    cnts = Counter(stats)
    with jsonlines.open(args.inputfile) as reader:
        for obj in reader:
        # Each obj is a dictionary representing a repository
            if obj['func_name']:
                cnts[stats_names[0]] += 1
                # counts newline characts just line `wc -l`
                cnts[stats_names[1]] += obj.get('original_string', "\n").count("\n") or 0
                cnts[stats_names[2]] += len(obj.get("docstring_tokens", [""])) or 0
                cnts[stats_names[3]] += len(obj.get("code_tokens", [""])) or 0
    ## now that we've summarized all the repos, print out the stats we care about
    ## note here we order by key name b/c the stats dict is unordered and not comparable across
    ## repos nor across statistics within a repo
    for key, cnt in sorted(cnts.items(), key= lambda x: x[0]):
        print(f"{key}: {cnt}")