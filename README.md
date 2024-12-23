
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

[Listings](vec_search/images/listing_landing_page.png)
Here you find the listings of all entries in your jsonl file
in an arbitrary order because no search has been performed.
Next use the search bar to enter a natural language search query.

[SERP](vec_search/images/search_results.png)
Here you will find the entities from your search
in semantic distance, closest is first and then proceeds
in ascending order.

[Inspection](vec_search/images/query_result_inspection.png)
After clicking the inspect button you will be taken to a
detail page where the dropdown menus provide a configurable
view of the attention weights from the semantic code search.
Hovering the mouse over code tokens or words enables inspection
of particular terms in the query or the code.

After sensemaking the human may want to iteratively modify
their natural language query using the 3 part workflow above.

# Roadmap Items

## Annotation workflow

The intent of this workflow is to enable a user to generate a benchmark or
golden dataset from their code-query corpus. To implement this workflow we
need:

i. A jsonl file containing the semantic index of the code & documentation, in
whichever languages.

ii. A query generated from the search textbar is persisted to disk.

iii. An annotation of the results provided.
  The user indicates which results-if any-are relevant to the query.
  The entries are persisted to a file on the local filesystem.
  After the user is done with the annotation effort, the system
  suggests to the user to upload the annotation data to a cloud object
  store to persist the data, multiple humans' annotations may eventually
  be merged for subsequent summarization of search performance for the corpus
  however, at the moment merging and analysis is not in scope.

The annotation workflow, and subsequently generated benchmark data are
useful for comparing a base model-say from huggingface-versus a fine tuned
model for the users needs or comparing two distinct models which are not fine
tuned.

## fine tune the model on custom data

TODO... design and write up the workflow

## compare two models (fine tuned or otherwise)

TODO... design and write up the workflow for data analysis
