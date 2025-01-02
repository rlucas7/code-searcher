
## make venv

```bash
python3 -m venv .
. bin/activate
pip install sqlite-vec sqlean.py flask click jsonlines
```

## initialize the db

```bash
flask --app vec_search init-db
```
Note that the above command populates the sqlite db with the vectors
from the given file.

## to run the app on localhost

```bash
flask --app vec_search run --debug
```

For the vectors we read these from a file at application startup time.
This allows for the vectors to be embedded from somewhere else. e.g.
any application-and any vector embedding method.

## execute a semantic search

You may have noticed the search form on the top of the listing page.
If you enter a natural language description like you'll see something like

```bash
"GET /?q=a+function+to+add+an+element+to+an+array HTTP/1.1" 200
```

in the stdout for the terminal where the flask app is running.
Until now we did not have embedding for the natural language text,
including the vector embeddings requires we install a few more ML dependencies.

```bash
pip install numpy==1.24 torch transformers
```

# execute searches locally

If you execute searches locally you probably want to include your own
index and not the current sampled jsonl file that contains only a few records.
We provide the sampled jsonl file to keep the size of the repo small whilst
maintaining a useable prototype.

# Inspect the SERPs

A search engine results page (SERP) shows the results from a natural
language query, in a format that is useful for the human.
For semantic search we use cosine distance between the query embedding
and the entities, in the example jsonl file provided these entities
are are programming language and associated documentation.

The embedding model for the query and the code provided is the codebert
model from Microsoft. This is to keep the model size small while still
working end to end. The code model is configurable via changes to the
 `config.py` module's `AI_MODEL` entry.
However, if you use a different model you may need to re-embed the code
entities to maintain proper alignment of vector spaces between query embeddings
and entity embeddings.

Typical inspection workflow:

![Listings](vec_search/images/listing_landing_page.png)
Here you find the listings of all entries in your jsonl file
in an arbitrary order because no search has been performed.
Next use the search bar to enter a natural language search query.

![SERP](vec_search/images/search_results.png)
Here you will find the entities from your search
in semantic distance, closest is first and then proceeds
in ascending order.

![Inspection](vec_search/images/query_result_inspection.png)
After clicking the inspect button you will be taken to a
detail page where the dropdown menus provide a configurable
view of the attention weights from the semantic code search.
Hovering the mouse over code tokens or words enables inspection
of particular terms in the query or the code.

After sensemaking the human may want to iteratively modify
their natural language query using the 3 part workflow above.

# Relevance Annotation

## Human workflow
The intent of this workflow is to enable a user to generate a benchmark AKA
golden dataset from their code-query corpus.

Purpose:
The annotation workflow, and subsequently generated benchmark data are
useful for comparing a base model-say from huggingface-versus a fine tuned
model for the users needs or comparing two distinct models which are not fine
tuned.

Steps:
1. Create-if necessary-and login as a user
2. Execute a query
3. The results of the query for the given corpus are shown in the SERP
4. Select relevance annotations for each result, clicking done once you
are sure of your relevance determination
5. Repeat steps 1-4 for each query

Notes:

The queries and query relevances are stored in tables
`queries` and `query_relevances` in the sqlite db.
Once the human has completed the annotations these may be exported from
the sqlite db for further processing.

The sqlite db file is located in the `var/vec_search-instance/` directory or more generally in the path specified in the `config.py` module for the application.

To collect the data for the annotations:

```bash
flask --app vec_search export-rad-to-csv rad.csv
```
Note that this exports to a file in the current working directory named
`rad.csv`. If you want a different filename this provide the alternate filename.
If the file already exists in the working directory then an overwrite will occur.I

## Manual workflow to generate relevance data

The click command is similar to this workflow:
```bash
# opens a REPL environment for sqlite3, if you modify the config.py then change the path
sqlite3  var/vec_search-instance/vec_search.sqlite

# make the field names displayed in results of queries and a comma separator
.mode column
.mode csv
# output results to csv file
.output relevance_annotation_details.csv

# annotation results
# we concatenate duplicates in a comma sep list (post_id, query_id, user_id)

SELECT
qr.query_id,
qr.post_id,
q.user_id,
GROUP_CONCAT(qr.relevance) AS relevances,
qr.rank,
qr.distance,
q.query
FROM query_relevances AS qr
INNER JOIN (
  SELECT query_id, query, user_id FROM queries
) AS q ON qr.query_id = q.query_id
GROUP BY
q.query_id,
user_id,
post_id
;

# exit sqlite REPL env
.quit
```

The file `relevance_annotation_details.csv` should contain the results of the above query.
This file is placed in the directory where you initiated the sqlite3 command.

For debugging purposes it is sometimes helpful
to see the schema for all tables in the REPL environment:

```bash
SELECT * FROM sqlite_master WHERE type='table';
```

# LLM annotation workflow (WIP-work in progress)

A backend/batch workflow where relevances are assessed outside of a human workflow
is the behavior currently supported.

There are 2 prompts in the `llm_rel_gen.py` module.

WIP:
- A relevance generation engine, really a thing wrapper around llm clients.

# Metrics generation (WIP-work in progress)

WIP:
- IR Retrieval metrics for the data once placed into pandas df(s).

## fine tune the model on custom data

TODO... design and write up the workflow

## compare two models (fine tuned or otherwise)

TODO... design and write up the workflow for data analysis
