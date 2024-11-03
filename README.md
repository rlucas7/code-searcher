
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
If you enter a natural language description you'll see something like

```bash
"GET /?q=an+array+destructor+that+leaves+the+contents+intact HTTP/1.1" 200
```

in the stdout for the terminal where the flask app is running.
Until now we did not have embedding for the natural language text,
including the vector embeddings requires we install a few more ML dependencies.

```bash
pip install numpy==1.24 torch transformers
```

