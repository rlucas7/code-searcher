"""
This module contains some common IR metrics
for evaluation of the relevance assessments done
by both humans and llms for search queries.

Because the `gen-llm-rels` command reads the input from
a csv generated by `export-rad-to-csv` command we assume all inputs
have that format and are subsequently stored in a
pandas dataframe.

The primary function to use from this module is `calc_ir_metrics` which
will calculate all the supported metrics.

Currently the following metrics are supported:

i. Cohen Kappa
ii. Spearman Correlation
iii. Kendall Tau
iv. Mean ave Precision@k
"""

from itertools import accumulate
from math import isclose

from flask import current_app as app
from scipy.stats import spearmanr, kendalltau, geom
from pandas import DataFrame, crosstab


def cohen_kappa(df: DataFrame) -> dict[str, float]:
    """Calculate Cohen Kappa for binary relevances.
    cf. https://nlp.stanford.edu/IR-book/pdf/08eval.pdf
    Args:
        df (DataFrame): A pandas dataframe that contains 2 binary relevance columns

    Returns:
        dict[str, float]: The calculated Cohen Kappa statistic
    """
    x_tab = crosstab(df.relevances, df.llm_rel_score, margins=True)
    # because of 0 on off diag we use a hacky good-turing smoother
    # TODO: make this work for ordinal >2 relevances
    # here for some queries if there are no 0-or no 1-then we raise key error and handle
    # TODO: make this cleaner code, but for now it seems to work on test cases...
    try:
        p_yes = x_tab[1][1]
    except KeyError:
        p_yes = 1
    try:
        p_no = x_tab[0][0]
    except KeyError:
        p_no = 1
    n_items = x_tab["All"]["All"]
    p_A = (p_yes + p_no) / n_items
    try:
        p_rel = x_tab[1]["All"]
    except KeyError:
        p_rel = 1
    try:
        p_rel += x_tab["All"][1]
    except KeyError:
        p_rel += 1
    p_rel /= 2 * n_items
    # now not rel...
    try:
        p_nrel = x_tab[0]["All"]
    except KeyError:
        p_nrel = 1
    try:
        p_nrel += x_tab["All"][0]
    except KeyError:
        p_nrel += 1
    p_nrel /= 2 * n_items
    p_E = p_rel * p_rel + p_nrel * p_nrel
    return {"Cohen-Kappa": (p_A - p_E) / (1.0 - p_E)}


def spearman_corr(df: DataFrame) -> dict[str, float]:
    """Calculate Spearman correlation

    Args:
        df (DataFrame):  A pandas dataframe that contains 2 relevance columns.

    Returns:
        dict[str, float]: Calculated correlation and corresponding p-value
    """
    s_corr, s_pv = spearmanr(df.relevances, df.llm_rel_score)
    return {"Spearman-corr": s_corr, "Spearman-p-value": s_pv}


def kendall_tau(df: DataFrame) -> dict[str, float]:
    """Calculate Kendall Tau

    Args:
        df (DataFrame): A pandas dataframe that contains 2 relevance columns.

    Returns:
        dict[str, float]: Calculated correlation and corresponding p-value
    """
    k_tau, k_pv = kendalltau(df.relevances, df.llm_rel_score)
    return {"Kendall-tau": k_tau, "Kendall-p-value": k_pv}


def mean_ave_prec(df: DataFrame, k: int = 10) -> dict[str, float]:
    """Computes the mean average precision at k.

    cf. https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
    Args:
        df (DataFrame): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        dict[str, float]: _description_
    """
    column_names = ["llm_rel_score", "relevances", "query_id", "rank"]
    # `grp` here is the query id and `adf` is the dataframe
    num_queries = 0.0
    # TODO: remove this `k` const and make in config
    map_at_k = 0.0
    offset = 1
    for grp, adf in df[column_names].groupby("query_id"):
        num_queries += 1.0
        print("\n\n", grp, "\n\n", adf)
        # converts ranks to a list-the ranks may have holes
        ranks = adf["rank"].values.tolist()
        relevances = adf["relevances"].values.tolist()
        rks = [0] * k
        rels = [0] * k
        # ranks are starting from 0, 1, ...
        for idx, r in enumerate(ranks):
            if r < k:
                rks[r] = 1 / (r + offset)
                if relevances[idx] == 1:
                    rels[r] = 1
        # these are the human rels
        csum_rels = list(accumulate(rels))
        num_possible = csum_rels[-1]  # total # of relevances
        prec_at_i = [csum_rels[idx] * rks[idx] * rels[idx] for idx in range(k)]
        ave_prec = sum(prec_at_i) / num_possible if num_possible > 0 else 0.0
        map_at_k += ave_prec
    # if there are no queries we should get 0.0
    return {"map_at_k": map_at_k / max(num_queries, 1)}


def rank_biased_overlap(df: DataFrame, k: int = 10, p: float = 0.1) -> dict[str, float]:
    """Calculates the Rank Biased Overlap (RBO) of two ranked lists.
    In the application here, one ranked list is generated by a human
    whilst the other ranked list is generated by an llm.

    Args:
        df (DataFrame): _description_
        k (int, optional): _description_. Defaults to 10.
        p (float, optional): The probability that a user will *not*
        persist searching once they've evaluated the given result.
        Defaults to 0.1.

    Returns:
        dict[str, float]: `'rbo@k', rbo-value` are the key/value pairs

    Notes: (i) The code assumes the probability is the same regardless
        of the value in the ranked results. This is primarily for mathematical
        simplification of closed form expressions.
        (ii) The code truncates at a given `k` and RBO is non-decreasing in `k`,
        e.g. RBO > RBB@k+1 >= RBO@k.
        (iii) RBO is not a metric but 1-RBO = RBD is, where the D is for distance,
        e.g. RBD is a distance metric so that the triangle inequality may be leveraged.

    References:
        A similarity measure for indefinite rankings by Webber, Moffat and Zobel
        2010, ACM Transactions on information systems.
    """
    column_names = ["llm_rel_score", "relevances", "query_id", "rank"]
    # `grp` is the query_id and `adf` is the data frame for the group
    num_queries = 0
    rbo = 0.0
    # NOTE: scipy.stats.geom has a reverse def of p, 1-p as compared to the RBO paper
    # in scipy the pmf is: `(1-p)^{k-1} p`` for k >= 1 whereas in the RBO paper the defn
    # is `w_d = (1-p)p^{k-1}`` so we swap p, with q = 1 - p
    persistence_prob = 1 - p
    g = geom(p=persistence_prob)
    rbo = 0.0
    for grp, adf in df[column_names].groupby("query_id"):
        # note that the ranks are sparse so we need to densify them for each query
        num_queries += 1
        llm_rels = [0] * k
        human_rels = [0] * k
        ranks = adf["rank"].values.tolist()
        hrel = adf["relevances"].values.tolist()
        lrel = adf["llm_rel_score"].values.tolist()
        # NOTE: the rob stat needs to start at 0 for each query
        rbo_stat = 0.0
        # the index in prefix overlap is offset by -1. e.g. `0 <= index < k`
        prefix_overlap = [0] * k
        for idx, r in enumerate(ranks):
            if r < k:
                llm_rels[r] = lrel[idx]
                human_rels[r] = hrel[idx]
        for idx, (human, llm) in enumerate(zip(human_rels, llm_rels)):
            if (human == 1 and llm == 1) or (human == 0 and llm == 0):
                rbo_stat += 1.0
            prefix_overlap[idx] = (g.pmf(idx + 1) * rbo_stat) / (idx + 1)
        rbo += sum(prefix_overlap)
    return {"rbo@k": rbo / num_queries}


def calc_ir_metrics(df: DataFrame, functions: list[callable] = []) -> dict[str, float]:
    """A wrapper around the metrics supported by the module.
    The intent is for this function to be used in the click command, `gen-ir-metrics`.

    Args:
        df (DataFrame): The dataframe to calculate metrics on.
        functions (list[callable], optional): A list of additional metrics to calculate.
            Defaults to [].

    Returns:
        dict[str, float]: A dictionary of the calculated metrics.
    """
    stats = {}
    funcs = [
        cohen_kappa,
        spearman_corr,
        kendall_tau,
        mean_ave_prec,
        rank_biased_overlap,
    ]
    if functions:
        funcs.extend(functions)
    for func in funcs:
        stats.update(func(df))
    return stats


if __name__ == "__main__":
    import unittest

    class TestMeanAveragePrecision(unittest.TestCase):
        def test_mapk_no_correct_should_be_0(self):
            df = DataFrame(
                {
                    "relevances": [0, 0, 0, 0],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [0, 0, 0, 0],
                }
            )
            assert mean_ave_prec(df)["map_at_k"] == 0.0

        def test_mapk_one_correct_equals_reciprocal_rank(self):
            df = DataFrame(
                {
                    "relevances": [0, 0, 1, 0],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [0, 0, 0, 0],
                }
            )
            assert mean_ave_prec(df)["map_at_k"] == 1 / 3

        def test_mapk_more_than_one_correct(self):
            df = DataFrame(
                {
                    "relevances": [1, 0, 1, 0],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [0, 0, 0, 0],
                }
            )
            # Note the 3rd entry must have a numerator of 2
            assert mean_ave_prec(df)["map_at_k"] == (1.0 + 2.0 / 3.0) / 2.0

    class TestRankBiasedOverlap(unittest.TestCase):
        def test_rbo_with_perfect_match_all_nonrelevant(self):
            df = DataFrame(
                {
                    "relevances": [0, 0, 0, 0],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [0, 0, 0, 0],
                }
            )
            assert isclose(rank_biased_overlap(df)["rbo@k"], 1.0)

        def test_rbo_with_perfect_match_all_relevant(self):
            df = DataFrame(
                {
                    "relevances": [1, 1, 1, 1],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [1, 1, 1, 1],
                }
            )
            assert isclose(rank_biased_overlap(df)["rbo@k"], 1.0)

        def test_rbo_with_no_match_but_some_relevant(self):
            df = DataFrame(
                {
                    "relevances": [1, 1, 0, 0],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [0, 0, 1, 1],
                }
            )
            # NOTE: here the value is ~ XXX e-5 so need to drop abs_tol default
            # a bit
            assert isclose(rank_biased_overlap(df)["rbo@k"], 0.0, abs_tol=0.0001)

        def test_rbo_with_all_match_but_some_relevant(self):
            df = DataFrame(
                {
                    "relevances": [1, 1, 0, 0],
                    "query_id": [1, 1, 1, 1],
                    "rank": [0, 1, 2, 3],
                    "llm_rel_score": [1, 1, 0, 0],
                }
            )
            # NOTE: here the value is ~ XXX e-5 so need to drop abs_tol default
            # a bit
            assert isclose(rank_biased_overlap(df)["rbo@k"], 1.0)

    # the main method here invokes discovery, execution, etc.
    unittest.main()
