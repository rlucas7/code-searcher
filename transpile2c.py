"""
This script transpiles the cosQA data to C using the `py2c` transpiler.
Approximately half of the examples from cosQA can be converted into
C-lang program code examples, demonstrating a scalable way to seed
a benchmark dataset in a new PL using an existing PL benchmark code search
data set.

Example invocation to summarize all transpiler errors:

```
python3 transpile2c.py --outputfile log.out --sample -1
```

If instead we want to see 1 or more but not all, say if you have a very large
benchmark dataset.

```
python3 transpile2c.py --outputfile log.out --sample 1
```

If you want to ignore the transpiler errors use

```
python3 transpile2c.py --outputfile log.out --sample 0
```
"""

import argparse
import csv
import sys

from collections import Counter
from json import loads
from random import sample, seed

from py2c.shortcuts import trans_c, trans_cpp
from py2c.exceptions import SourceCodeException
from requests import get as RequestsGet



parser = argparse.ArgumentParser(description="A simple script to transpile python to C.")

parser.add_argument("--outputfile", type=str, help="The file name where the transpiled code is written.")
parser.add_argument("--seed", type=int, help="The seed integer for random sampling reproducibility.", default=42)
parser.add_argument("--samples", type=int, help="The number of samples for random sampling of transpiler exceptions. Use 0 or less for all exceptions.", default=30)
parser.add_argument("--verbose", action="store_true", help="Verbose output, shows full source code in addition to transpiler exception.")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    data_url = "https://raw.githubusercontent.com/Jun-jie-Huang/CoCLR/refs/heads/main/data/search/cosqa-retrieval-train-19604.json"
    resp = RequestsGet(data_url)
    assert resp.status_code==200, f"Got an error code from Github: {resp.status_code}, maybe .. try again?"
    contents = loads(resp.content)
    N = len(contents)
    transpiled_contents = []
    exceptions_indices = []
    for idx in range(N):
        try:
            val = dict()
            val["query_id"] = idx
            val["post_id"] = 1
            val["user_id"] = 1
            val["relevances"] = contents[idx]['label']
            val["rank"] = 1
            val["distance"] = 1
            val['query'] = contents[idx]['doc'].replace('python', 'c')
            val['doc'] = "None"
            val['code'] = trans_c(contents[idx]['code'])
            transpiled_contents.append(val)
        except Exception as e:
            if args.verbose:
                print(f"exception at index: {idx}...")
            exceptions_indices.append((idx, e))
    # approx 50% can be transpiled directly
    # out of the 10,583 / 19,604 = 0.5398388084064477
    #
    # the errors are largely non-compatibility issues, e.g. no exceptions in C, no list nor generator comprehensions etc.
    # this is basically the same for c or cpp transpiler
    # then for these successfully transpiled, we do a RAD pass...
    with open(args.outputfile, "w", newline="\n") as csv_file:
        fieldnames = [
            "query_id",
            "post_id",
            "user_id",
            "relevances",
            "rank",
            "distance",
            "query",
            "doc",
            "code",
        ]
        dw = csv.DictWriter(
            csv_file,
            delimiter="|",
            quotechar='"',
            fieldnames=fieldnames,
            lineterminator="\r\n",
        )
        dw.writeheader()
        for i, record in enumerate(transpiled_contents):
            dw.writerow(record)

    ## quantify the error typed in the transpilation process...
    print("==============Summary of examples leading to exceptions in transpilation==============")
    exception_cnts = Counter([type(e[1]) for e in exceptions_indices])
    ttl = sum(exception_cnts.values())
    print("error, total #, percentage-ttl")
    for key, cnt in sorted(exception_cnts.items(), key=lambda x: -x[1]):
        print(key, cnt,  100*round(cnt/ttl, 3))

    ## 70.00% of the issues are py2c.exceptions.SourceCodeException
    ## 12.90% are a generic exception
    ## 08.00% py2c.exceptions.NoneIsNotAllowedException
    ## 00.90% syntax error
    ## 05.90% are invalid annotations
    ## 02.90% attribute error

    ## now looking at the largest source we sample X of these and inspect manually
    src_excs = [(idx, exc) for idx, exc in exceptions_indices if isinstance(exc, SourceCodeException)]
    if args.samples > 0:
        if args.samples > len(src_excs):
            print(f"Error: --sample ({args.samples}) is greater than the number of available SourceCodeException examples ({len(src_excs)}).")
            sys.exit(1)
        seed(args.seed)
        samp = sample(src_excs, args.samples)
        src_code_excs = Counter()
        for s in samp:
            if args.verbose:
                print(s)
                print(contents[s[0]]['code'])
            ## some automatic line of code error determinations...
            line, node_name = s[1].args[0].split("Line:")[-1].split("Name:")
            line.strip(), node_name.strip()
            src_code_excs[node_name.strip()] += 1
            if line.strip().split("/")[0] != "None":
                line_no = int(line.strip().split("/")[0]) - 1
                if args.verbose:
                    print("e.g. ---->", contents[s[0]]['code'].split("\n")[line_no])
            else:
                if args.verbose:
                    print("line no not tracked...")
    elif args.samples < 0:
        src_code_excs = Counter()
        for s in src_excs:
            if args.verbose:
                print(s)
                print(contents[s[0]]['code'])
            ## some automatic line of code error determinations...
            line, node_name = s[1].args[0].split("Line:")[-1].split("Name:")
            line.strip(), node_name.strip()
            src_code_excs[node_name.strip()] += 1
            if line.strip().split("/")[0] != "None":
                line_no = int(line.strip().split("/")[0]) - 1
                if args.verbose:
                    print("e.g. ---->", contents[s[0]]['code'].split("\n")[line_no])
            else:
                if args.verbose:
                    print("line no not tracked...")
    else:
        sys.exit(0)
    # now display the results of transpiler errors
    ttl_src_exc = sum(src_code_excs.values())
    # print in descending order
    print("==============Summary of language differences leading to exceptions in transpilation==============")
    print("error, total #, percentage-ttl")
    for key, cnt in sorted(src_code_excs.items(), key=lambda x: -x[1]):
        print(key, cnt,  round(100*round(cnt/ttl_src_exc, 3),1))