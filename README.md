
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

## to run the app on localhost

```bash
flask --app vec_search run --debug
```
