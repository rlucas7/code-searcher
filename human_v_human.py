# some stuff related to comparing annotations of the two humans
import argparse

from numpy import diag
from pandas import read_csv, concat, merge, crosstab


def accuracy(crosstab, margin:str):
    diagonal_sum = diag(crosstab).sum()
    total_sum = crosstab[margin][margin] # assumes this name
    # diagonal sum includes total so we remove to correctly calculate
    return (diagonal_sum - total_sum) / total_sum

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefix', type=str, default="backups/radzz/")
parser.add_argument('--lang', type=str, default="c", choices=["c", "go", "js", "java", "python"])
parser.add_argument('--retriever', type=str, default="bm25", choices=["bm25", "codet5p", "codebert"])
parser.add_argument('--margin_name', type=str, default='Grand Total')

if __name__ == "__main__":
    args = parser.parse_args()
    # cli inputs...
    file1 = f"rad-{args.lang}-lang-D-{args.retriever}.csv"
    file2 = f"rad-{args.lang}-lang-lrr-{args.retriever}.csv"
    filename1 = args.prefix + file1
    filename2 = args.prefix + file2

    # this lambda takes the final selected relevance for the query & post id
    func = lambda x: x.split(",").pop()
    convs = {"relevances": func}
    usecols = [
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
    df_h1 = read_csv(filename1, sep="|", header=0, converters=convs, usecols=usecols)
    df_h2 = read_csv(filename2, sep="|", header=0, converters=convs, usecols=usecols)
    m_df = merge(df_h1, df_h2, on=['query', 'post_id', 'rank'], how='inner')
    x_tab = crosstab(m_df['relevances_x'], m_df['relevances_y'], margins=True, margins_name=args.margin_name)
    # pipe this properly...
    print(x_tab)
    print(accuracy(x_tab, args.margin_name))
