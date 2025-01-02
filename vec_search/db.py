import csv
import json
import sqlean as sqlite3
import sqlite_vec
from datetime import datetime

import click
import jsonlines
from flask import current_app, g
from pandas import read_csv

# HACK
from vec_search.config import _SQLITE_VEC_DLL_PATH, _JSONL_LOCAL_FILE


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.enable_load_extension(True)
        g.db.load_extension(_SQLITE_VEC_DLL_PATH)
        sqlite_vec.load(g.db)
        g.db.enable_load_extension(False)
        # NOTE: makes each row mutable, whereas  sqlite.db2.Row is not
        g.db.row_factory = dict_factory
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.enable_load_extension(True)
    db.load_extension(_SQLITE_VEC_DLL_PATH)
    sqlite_vec.load(g.db)
    db.enable_load_extension(False)

    #TODO: make this a log instead
    vec_version = db.execute("select vec_version()").fetchone()
    print(f"vec_version={vec_version}")

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

    # insert the vectors
    DDL_vec_insert_cmd = """insert into vec_items(rowid, embedding)
        values (?, ?)
    """

    # insert the content associated with vectors
    DDL_content_insert_cmd = """ insert into post(id, func_name, path, sha, code, doc)
        values (?, ?, ?, ?, ?, ?)
    """
    names = set()
    with jsonlines.open(_JSONL_LOCAL_FILE) as reader:
        for idx, obj in enumerate(reader, start=1):
            jl = json.loads(obj)
            if not jl['func_name'] in names:
                names.add(jl['func_name'])
                db.execute(DDL_vec_insert_cmd, [idx, sqlite_vec.serialize_float32(jl['embeddings'])])
                db.execute(DDL_content_insert_cmd, [idx, jl['func_name'], jl['path'], jl['sha'], jl['original_string'], jl['docstring']])
            else:
                print(f"{idx}th record is a duplicate, at path: {jl['path']}, function name: {jl['func_name']}")
    db.commit()

    # now confirm the vectors are loaded to the sqlite db instance in flask
    query = """select vec_to_json(vec_slice(embedding, 0, 8)) from vec_items limit 10"""
    cur = db.cursor()
    cur.execute(query)
    all_rows = cur.fetchall()
    for v in all_rows:
        print(v)

    vals = cur.execute("select count(*) from vec_items").fetchall()
    print(f"total number of vectors stored is: {vals}")
    ps = cur.execute("select count(*) from post").fetchall()
    print(f"total number of associated entries stored is: {ps}")
    one = cur.execute("select * from post limit 1").fetchall()
    for v in one:
        for idx, e in enumerate(v):
            print(idx, e)
    # display to stdout the metadata for tables ...
    one = cur.execute("SELECT * FROM sqlite_master WHERE type='table';").fetchall()
    for v in one:
        print(v)
    cur.close()
    db.close()


@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


@click.command('export-rad-to-csv')
@click.argument('filename', type=click.Path(exists=False))
def export_rad_to_csv(filename):
    click.echo("Now processing")
    click.echo(click.format_filename(filename))
    db = sqlite3.connect(
        current_app.config['DATABASE'],
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    db.row_factory = dict_factory
    with open('vec_search/RAD.sql', 'r') as f:
        query = f.read()
    with open(filename, 'w', newline='\n') as csv_file:
        fieldnames = ['query_id', 'post_id', 'user_id', 'relevances', 'rank', 'distance', 'query']
        dw = csv.DictWriter(csv_file, delimiter='|', quotechar='[', fieldnames=fieldnames)
        dw.writeheader()
        for row in db.execute(query).fetchall():
            dw.writerow(row)
    click.echo(f"relevance results written to {click.format_filename(filename)}")

sqlite3.register_converter(
    "timestamp", lambda v: datetime.fromisoformat(v.decode())
)

@click.command('gen-llm-rels')
@click.argument('filename', type=click.Path(exists=True))
@click.argument('dupstrat', type=str, default='takelast')
def gen_llm_rels(filename, dupstrat):
    ## NOTE:
    # 1. this assumes a file in the format exported by the export click command above
    # has been executed locally
    # because there may be vacillation on the part of the human we need a
    # converter for the read in the event that the relevance column has multiple
    # entries, these are stored as `|0,1,...|` with additional binary relevances replacing
    # the ellipses, if there are no relevances for a row then an error is thrown
    # 2. if 'takelast' is set then pop method takes the last assigned relevance if there are > 1 for the record
    if dupstrat == 'takelast':
        func = lambda x: x.split(",").pop()
    else:
        raise ValueError("error: currently only 'takelast' is supported for dupstrat...")
    convs = {"relevances": func}
    # rows2skip = [4] # use with skiprows=
    usecols = ["query_id", "post_id", "user_id", "rank", "distance", "query", "relevances"]
    df = read_csv(filename, sep='|', header=0, converters=convs, usecols=usecols)
    ## now the df has been read in and we want to generate the IR metrics...
    print("all done...")

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(export_rad_to_csv)
    app.cli.add_command(gen_llm_rels)
