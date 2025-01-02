"""
This module contains some common IR metrics
for evaluation of the relevance assessments done
by both humans and llms for search queries.

Because the `gen-llm-rels` command reads the input from
a csv generated by `export-rad-to-csv` command we assume all inputs
have that format and are subsequently stored in a
pandas dataframe.
"""



