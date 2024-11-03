
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
