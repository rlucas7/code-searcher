# requires an install from github via:
# pip install git+https://github.com/syeysk/sy_py2c.git
# and then you can uninstall after using, if you want via
# pip uninstall py2c
import argparse
import csv

from py2c.shortcuts import trans_c, trans_cpp
from requests import get as RequestsGet
from json import loads

parser = argparse.ArgumentParser(description="A simple script to transpile python to C.")

parser.add_argument("--outputfile", type=str, help="The file name where the transpiled code is written.")


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

