import json
import sqlean as sqlite3
import sqlite_vec
from datetime import datetime

import click
import jsonlines
from flask import current_app, g


#TODO: put this into config and read via config parser
_SQLITE_VEC_DDL_PATH = "/Users/rlucas/sqlite-ext/sqlite-vec/dist/vec0.dylib"
_JSONL_LOCAL_FILE = "collections-c-embeds.jsonl"


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.enable_load_extension(True)
        #g.db.load_extension(_SQLITE_VEC_DDL_PATH)
        sqlite_vec.load(g.db)
        g.db.enable_load_extension(False)
        g.db.row_factory = sqlite3.Row

    ps = g.db.cursor().execute("select count(*) from post").fetchall()
    print(f"total number of associated entires stored is: {ps[0][0]}")

    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.enable_load_extension(True)
    # db.load_extension(_SQLITE_VEC_DDL_PATH)
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
    DDL_content_insert_cmd = """ insert into post(id, func_name, path, sha)
        values (?, ?, ?, ?)
    """
    names = set()
    with jsonlines.open(_JSONL_LOCAL_FILE) as reader:
        for idx, obj in enumerate(reader, start=1):
            jl = json.loads(obj)
            if not jl['func_name'] in names:
                names.add(jl['func_name'])
                db.execute(DDL_vec_insert_cmd, [idx, sqlite_vec.serialize_float32(jl['embeddings'])])
                db.execute(DDL_content_insert_cmd, [idx, jl['func_name'], jl['path'], jl['sha']])
            else:
                print(f"{idx}th record is a duplicate, at path: {jl['path']}, function name: {jl['func_name']}")

    # now confirm the vectors are loaded to the sqlite db instance in flask
    query = """select vec_to_json(vec_slice(embedding, 0, 8)) from vec_items limit 10"""
    cur = db.cursor()
    cur.execute(query)
    all_rows = cur.fetchall()
    for v in all_rows:
        print(v[0])

    vals = cur.execute("select count(*) from vec_items").fetchall()
    print(f"total number of vectors stored is: {vals[0][0]}")
    ps = cur.execute("select count(*) from post").fetchall()
    print(f"total number of associated entires stored is: {ps[0][0]}")
    one = cur.execute("select * from post limit 1").fetchall()
    for v in one:
        for idx, e in enumerate(v):
            print(idx, e)

@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


sqlite3.register_converter(
    "timestamp", lambda v: datetime.fromisoformat(v.decode())
)

def init_app(app):

    #DDL_create_cmd = """create virtual table vec_items using
    #    vec0(contents_embedding float[768] distance_metric=cosine)
    #"""
    #db_inst.execute(DDL_create_cmd)


    # now the typical items...
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
