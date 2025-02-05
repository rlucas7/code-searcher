import sys

import sqlite_vec
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from torch import tensor, reshape, FloatTensor
from werkzeug.exceptions import abort

from bertviz.transformers_neuron_view import RobertaModel as RM, RobertaTokenizer as RT
from bertviz.head_view import head_view
from bertviz.neuron_view import show, get_attention
from markupsafe import Markup

from vec_search.auth import login_required
from vec_search.db import get_db

# HACK: for now hard codes the location of the config file for AI MODEL
from vec_search.config import AI_MODEL as MODEL

bp = Blueprint("search", __name__)


if str(sys.argv[3]) == "run":
    # If other commands are added to the app that need access to
    # the LLM then they need to be added here. Currenly using 'run'
    # in this case-and only this case for now-we import the HF LLM
    # the basic purpose here is to not do a reload for every invocation
    # of a cli cmd
    from transformers import RobertaTokenizer, RobertaForMaskedLM
    from vec_search.config import AI_MODEL as MODEL
    _MODEL = RobertaForMaskedLM.from_pretrained(MODEL)
    _TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)


# we hack the GET & disambiguate a search
@bp.route("/", methods=["GET"])
def index():
    db = get_db()
    if request.method == "GET" and request.args.get("q") is None:
        # TODO: setup pagination
        posts = db.execute(
            "SELECT func_name, path, sha, code, doc FROM post limit 10"
        ).fetchall()
        return render_template("search/index.html", posts=posts)
    elif request.method == "GET" and request.args.get("q") is not None:
        # load the LLM to embed the natural lang text to a vec
        q = request.args.get("q")
        if g.user is not None:
            # if user is logged in then record the query to query table
            SAVE_QUERY_CMD = "INSERT INTO queries(query, user_id) values (?, ?)"
            db.execute(SAVE_QUERY_CMD, [q, g.user["id"]])
            db.commit()
        # block of code to embed natural language query
        tokens = (
            [_TOKENIZER.cls_token] + _TOKENIZER.tokenize(q) + [_TOKENIZER.eos_token]
        )
        raw_token_ids = tensor(_TOKENIZER.convert_tokens_to_ids(tokens))
        tokens_ids = reshape(raw_token_ids, (1, len(raw_token_ids)))
        context_embeddings = _MODEL(tokens_ids, output_hidden_states=True)
        embeds = context_embeddings.hidden_states[-1].detach().numpy()[0, 0, :]
        embed = embeds.tolist()
        # find matches
        cur = db.cursor()
        vec_query = """
            with knn_matches as (
              select
                rowid,
                distance
              from vec_items
              where embedding match ?
                and k = 10
            )
            select
              func_name,
              path,
              sha,
              code,
              doc,
              knn_matches.distance as distance,
              post.id as postid
            from knn_matches
            left join post on post.id = knn_matches.rowid
        """
        # grab the query id to keep the ids client unique
        for row in db.execute("SELECT last_insert_rowid()").fetchall():
            query_id = row["last_insert_rowid()"]
        cur.execute(vec_query, [sqlite_vec.serialize_float32(embed)])
        posts = cur.fetchall()
        for i in range(len(posts)):
            posts[i].update({"search-query": q})
            if g.user and q:
                posts[i].update(
                    {"query-id": str(query_id) + f"+post-num-{i}", "rank": str(i)}
                )
        cur.close()
        return render_template("search/index.html", posts=posts)
    else:
        raise ValueError("should not ever enter this branch")


@bp.route("/detail", methods=["GET"])
def detail():
    entity_id = request.args.get("postid")
    search_query = request.args.get("query")
    cross_type = request.args.get("cross_type", "NL")
    post = (
        get_db()
        .execute(
            "SELECT p.id, code, doc" " FROM post p" " WHERE p.id = ?", (entity_id,)
        )
        .fetchone()
    )
    # Now we construct a string of the query + post-id to feed through the model
    # and get the cross attentions (query -> post-id) for rendering visualiztion
    # in the browser. This uses BertViz-which relies on d3.js
    model_version = "roberta-base"
    model = RM.from_pretrained(model_version, output_attentions=True)
    tokenizer = RT.from_pretrained(model_version, do_lower_case=True)
    sentence1 = search_query
    # the viz tends to be too long for the screen with both NL + PL so we branch
    if cross_type == "NL":
        # NL = natural lang
        sentence2 = post["doc"]
    elif cross_type == "PL":
        sentence2 = post["code"]
    else:
        raise ValueError("unsupported cross attention type requested")
    # the head view is only impl for limitations of bertviz
    attn_data = get_attention(
        model,
        "roberta",
        tokenizer,
        sentence1,
        sentence2,
        include_queries_and_keys=False,
    )
    # annoyingly `tokens` seems is a required albeit redundant arg for head_view
    # some hacky stuff to get things to match the signature of the `head_view()` function ...
    tokens = attn_data["ba"]["right_text"] + attn_data["ba"]["left_text"]
    sentence_2_start = len(attn_data["ba"]["right_text"])
    attention = []
    n = len(attn_data["all"]["attn"])
    for i in range(n):
        attention.append(FloatTensor(attn_data["all"]["attn"][i]).unsqueeze(0))

    ret_obj = head_view(
        attention, tokens, sentence_2_start, include_layers=[11], html_action="return"
    )
    content = {"script": Markup(ret_obj.data), "id": post["id"], "type": cross_type}
    return render_template("search/detail.html", content=content)


# code for relevance assessment workflow
@bp.route("/relevance", methods=["GET"])
def relevance():
    db = get_db()
    post_id = int(request.args.get("post-id"))
    query_id = int(request.args.get("query-id"))
    rel = 1 if request.args.get("relevance") == "yes" else 0
    rank = int(request.args.get("rank"))
    dist = float(request.args.get("distance"))
    rel_record = [post_id, query_id, rel, rank, dist]
    # store the relevance data ...
    # NOTE: the relevance table has no constraint on (post_id, query_id)
    # being unique (to not break workflow of human relevance change yes->no or vice versa)
    # however when making a final dataset you would likely want to make
    # (post_id, query_id) unique as the primary key...
    SAVE_RELEVANCE_CMD = "INSERT INTO query_relevances(post_id, query_id, relevance, rank, distance) values (?, ?, ?, ?, ?)"
    db.execute(SAVE_RELEVANCE_CMD, rel_record)
    db.commit()
    # NOTE: we expect no returning content from this endpoint but we do
    # send back a simple status to ACK ...
    return {"status": "ok"}


def get_post(id, check_author=True):
    post = (
        get_db()
        .execute(
            "SELECT p.id, func_name, path, sha" " FROM post p" " WHERE p.id = ?", (id,)
        )
        .fetchone()
    )

    if post is None:
        abort(404, f"Post id {id} doesn't exist.")

    if check_author and post["author_id"] != g.user["id"]:
        abort(403)

    return post


# NOTE: this is not fully implemented
@bp.route("/<int:id>/update", methods=("GET", "POST"))
@login_required
def update(id):
    post = get_post(id)

    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        error = None

        if not title:
            error = "Title is required."

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                "UPDATE post SET title = ?, body = ?" " WHERE id = ?", (title, body, id)
            )
            db.commit()
            return redirect(url_for("search.index"))

    return render_template("search/update.html", post=post)
