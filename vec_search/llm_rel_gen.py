"""
This module houses the framework for generating LLM relevances for search queries.
"""

# given in https://arxiv.org/pdf/2406.06519 fig 1
umb_promt = """
Given a query and a passage, you must provide a score on an
integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but
does not answer it,
2 = represents that the passage has some answer for the query,
but the answer may be a bit unclear, or hidden amongst extraneous
information and
3 = represents that the passage is dedicated to the query and
contains the exact answer.
Important Instruction: Assign category 1 if the passage is
somewhat related to the topic but not completely, category 2 if
passage presents something very important related to the entire
topic but also has some extra information and category 3 if the
passage only and entirely refers to the topic. If none of the
above satisfies give it category 0.
Query: {query}
Passage: {passage}
Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query
(M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each,
and decide on a final score (O). Final score must be an integer
value only.
Do not provide any code in result. Provide each score in the
format of: ##final score: score without providing any reasoning.
"""

llm_rel_assess = """
You are a search quality rater evaluating the relevance
of web pages. Given a query and a web page, you must
provide a score on an integer scale of 0 to 2 with the
following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain
other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the
topic. If you would use any of the information contained
in the web page in such a report, mark it 1. If the web page
is primarily about the topic, or contains vital information
about the topic, mark it 2. Otherwise, mark it 0.

Query
A person has typed
{query}
into a search engine.

They were looking for:
{narrative}

Result
Consider the following web page.
—BEGIN CONTENT—
{passage}
—END CONTENT—
Instructions
Split this problem into steps:

Measure how well the content matches a likely intent of
the query (M).

Measure how trustworthy the web page is (T).

Consider the aspects above and the relative importance
of each, and decide on a final score (O).

Produce a JSON array of scores without providing any
reasoning. Example: [{"M": 2, "T": 1, "O": 1}, {"M":
1 . . .
Results
[{
"""

# Note the narrative for the second prompt is unlikely to be provided
# given that we do not have a browser worflow interface setup for this part-yet.

# TODO: make the narrative optional
# TODO: implement the narrative input workflow