import csv
import sqlean as sqlite3
import sqlite_vec
from datetime import datetime

import click
import jsonlines
from flask import current_app, g
from pandas import read_csv, concat

# NOTE these will need to change when config.py's AI_MODEL changes

# HACK
from vec_search.config import _SQLITE_VEC_DLL_PATH, _JSONL_LOCAL_FILE, VEC_DIM, EMBED_ON_LOAD, AI_MODEL
from vec_search.retriever import Embedder
from .llm_rel_gen import LLMRelAssessor, Prompt, _umb_promt
from .ir_eval_metrics import calc_ir_metrics


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(
            current_app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.enable_load_extension(True)
        g.db.load_extension(_SQLITE_VEC_DLL_PATH)
        sqlite_vec.load(g.db)
        g.db.enable_load_extension(False)
        # NOTE: makes each row mutable, whereas  sqlite.db2.Row is not
        g.db.row_factory = dict_factory
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.enable_load_extension(True)
    db.load_extension(_SQLITE_VEC_DLL_PATH)
    sqlite_vec.load(g.db)
    db.enable_load_extension(False)

    # TODO: make this a log instead
    vec_version = db.execute("select vec_version()").fetchone()
    current_app.logger.info(f"vec_version={vec_version}")

    # NOTE: here schema[:517] + "???" + schema[520:] will put triple
    # question marks for the vector dimension of the `vec_items` table
    # and in general we want this value to be the vector dimension from
    # the `config.py` file. If you change the contents of `schema.sql`
    # then the logic here breaks if changes occur before character point
    # 519, so avoid that if at all possible. If the dimensions don't match
    # then you'll get a somewhat informative error on what the system
    # expected for vector dimension.
    with current_app.open_resource("schema.sql") as f:
        schema = f.read().decode("utf8")
        script = schema[:517] + str(VEC_DIM) + schema[520:]
        db.executescript(script)

    # insert the vectors
    DDL_vec_insert_cmd = """insert into vec_items(rowid, embedding)
        values (?, ?)
    """

    # insert the content associated with vectors
    DDL_content_insert_cmd = """ insert into post(id, func_name, path, sha, code, doc)
        values (?, ?, ?, ?, ?, ?)
    """
    embedder = Embedder(hf_model=AI_MODEL)
    with jsonlines.open(_JSONL_LOCAL_FILE) as reader:
        current_app.logger.info("loading contents into db...")
        for idx, obj in enumerate(reader, start=1):
            if EMBED_ON_LOAD:
                embed = embedder.embed(obj)
            else:
                # here we load the pre-embedded (using codebert unless updated)
                embed = obj["embeddings"]
            db.execute(
                DDL_vec_insert_cmd,
                [idx, sqlite_vec.serialize_float32(embed)],
            )
            db.execute(
                DDL_content_insert_cmd,
                [
                    idx,
                    obj["func_name"],
                    obj["path"],
                    obj["sha"],
                    obj["original_string"],
                    obj["docstring"],
                ],
            )
    db.commit()

    # now confirm the vectors are loaded to the sqlite db instance in flask
    query = """select vec_to_json(vec_slice(embedding, 0, 8)) from vec_items limit 3"""
    cur = db.cursor()
    cur.execute(query)
    all_rows = cur.fetchall()
    for v in all_rows:
        current_app.logger.info(str(v))

    vals = cur.execute("select count(*) from vec_items").fetchall()
    current_app.logger.info(f"total number of vectors stored is: {str(vals)}")
    ps = cur.execute("select count(*) from post").fetchall()
    current_app.logger.info(f"total number of associated entries stored is: {str(ps)}")
    one = cur.execute("select * from post limit 1").fetchall()
    for v in one:
        for idx, e in enumerate(v):
            current_app.logger.info(str(idx) + "   "+ str(e))
    # display to stdout the metadata for tables ...
    one = cur.execute("SELECT * FROM sqlite_master WHERE type='table';").fetchall()
    for v in one:
        current_app.logger.info(str(v))
    cur.close()
    db.close()


@click.command("reset-users")
def reset_users():
    drop_sql_cmd = "DROP TABLE IF EXISTS user;"
    create_sql_cmd = "CREATE TABLE IF NOT EXISTS user (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL);"
    db = get_db()
    db.execute(drop_sql_cmd)
    db.commit()
    db.execute(create_sql_cmd)
    db.commit()
    click.echo(
        "WARNING: the user table has been reset, all user_id associations in the queries table are lost ..."
    )
    db.execute(
        "INSERT INTO user (username, password) VALUES (?, ?)",
            (-1, "anonymous"),
    )
    db.commit()
    click.echo(
        "INFO: the user table has the anonymous user added, all non-authenticated user_id associations in the queries table are associated to this user"
    )

@click.command("init-db")
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo("Initialized the database.")


@click.command("export-rad-to-csv")
@click.argument("filename", type=click.Path(exists=False))
def export_rad_to_csv(filename):
    click.echo("Now processing")
    click.echo(click.format_filename(filename))
    db = sqlite3.connect(
        current_app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
    )
    db.row_factory = dict_factory
    with open("vec_search/RAD.sql", "r") as f:
        query = f.read()
    with open(filename, "w", newline="\n") as csv_file:
        fieldnames = [
            "query_id",
            "post_id",
            "user_id",
            "relevances",
            "rank",
            "distance",
            "query",
            "doc",
            "code",
        ]
        dw = csv.DictWriter(
            csv_file,
            delimiter="|",
            quotechar='"',
            fieldnames=fieldnames,
            lineterminator="\r\n",
        )
        dw.writeheader()
        for row in db.execute(query).fetchall():
            dw.writerow(row)
    click.echo(f"relevance results written to {click.format_filename(filename)}")



sqlite3.register_converter("timestamp", lambda v: datetime.fromisoformat(v.decode()))


@click.command('gen-llm-rels')
@click.argument('filename', type=click.Path(exists=True))
@click.argument('output_filename', type=click.Path(exists=False))
@click.argument('llm_model', type=click.Choice(['openai', 'gemini', 'aws', 'llama4']))
@click.argument('dupstrat', type=str, default='takelast')
def gen_llm_rels(filename, output_filename, llm_model, dupstrat):
    ## NOTE:
    # 1. this assumes a file in the format exported by the export click command above
    # has been executed locally
    # because there may be vacillation by the human the relevance column may have multiple
    # entries, these are stored as `|0,1,...|`
    if dupstrat == "takelast":
        # NOTE: you may want to vary duplicate handling strategies here
        func = lambda x: x.split(",").pop()
    else:
        raise ValueError(
            "error: currently only 'takelast' is supported for dupstrat..."
        )
    convs = {"relevances": func}
    # rows2skip = [4] # use with skiprows=
    usecols = [
        "query_id",
        "post_id",
        "user_id",
        "relevances",
        "rank",
        "distance",
        "query",
        "doc",
        "code",
    ]
    df = read_csv(filename, sep="|", header=0, converters=convs, usecols=usecols)
    click.echo(df.head())
    click.echo(df.shape)
    prompt = Prompt(_umb_promt)
    llm_rel = LLMRelAssessor(df, output_filename, prompt=prompt, model_name=llm_model)
    llm_rel.generate_rel(parse=True)
    click.echo("all done...")


# NOTE: while we could invole `gen-llm-rels` inside this cmd
# clicks in clicks are discouraged, for more:
# https://click.palletsprojects.com/en/stable/advanced/#invoking-other-commands
# plus this gives us a better workflow for inspecting intermediate outputs
@click.command("gen-ir-metrics")
@click.argument("filename", type=click.Path(exists=True))
def gen_ir_metrics(filename):
    # we assume input is the output of `gen-llm-rels`
    df = read_csv(filename)
    stats = calc_ir_metrics(df)
    # print all the metrics
    for k, v in stats.items():
        click.echo(f"{k}: {v}")


# NOTE: `nargs=-1` indicates an arbitrary number
# we expect at least 2
@click.command("rad-merge", context_settings={"ignore_unknown_options": True})
@click.argument("filenames", nargs=-1, type=click.Path(exists=True))
@click.argument("output_filename", type=click.Path(exists=False))
def rad_merge(filenames, output_filename):
    # we assume input is the output of `gen-llm-rels` so they're in that format
    assert (
        len(filenames) >= 2
    ), f"expected at least 2 files but got {len(filenames)} ..."
    # setup DF
    usecols = [
        "query_id",
        "post_id",
        "user_id",
        "relevances",
        "rank",
        "distance",
        "query",
        "doc",
        "code",
    ]
    func = lambda x: x.split(",").pop()
    convs = {"relevances": func}
    click.echo(f"basing dataframe off of {click.format_filename(filenames[0])} ...")
    dfs = [read_csv(filenames[0], sep="|", header=0, converters=convs, usecols=usecols)]
    for filename in filenames[1:]:
        click.echo(f"merging dataframes: {click.format_filename(filename)} ...")
        dfs.append(
            read_csv(filename, sep="|", header=0, converters=convs, usecols=usecols)
        )
    df = concat(dfs, axis=0)
    click.echo(df.head())
    click.echo(df.shape)
    click.echo(
        f"writing merged dataframes: {click.format_filename(output_filename)} ..."
    )
    df.to_csv(
        output_filename, sep="|", quotechar='"', columns=usecols, lineterminator="\r\n"
    )
    click.echo("all done ...")


@click.command('docs-cov-top-level')
@click.argument('filenames', nargs=-1, type=click.Path(exists=True))
def docs_cov_top_level(filenames):
    # for a given indexed repo determine how much indexed code has docs
    for file in filenames:
        no_docs_cnt, ttl_entities = 0, 0
        click.echo(f"Processing file: {file}")
        with jsonlines.open(file) as reader:
            for _, obj in enumerate(reader, start=1):
                # NOTE: in the indexing code I use this string literal if no docs
                if obj['docstring'] == 'NO-DOCs':
                    no_docs_cnt +=1
                ttl_entities += 1
        click.echo(f"Total Function Declarations: {ttl_entities}")
        click.echo(f"Total Function Declarations without docs: {no_docs_cnt}")
        click.echo(f"Proportion without docs: {round(100*no_docs_cnt/ttl_entities, 2)}%")


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(export_rad_to_csv)
    app.cli.add_command(gen_llm_rels)
    app.cli.add_command(gen_ir_metrics)
    app.cli.add_command(reset_users)
    app.cli.add_command(rad_merge)
    app.cli.add_command(docs_cov_top_level)
