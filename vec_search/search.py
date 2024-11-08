import sqlite_vec
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from torch import tensor, reshape
from transformers import RobertaTokenizer, RobertaForMaskedLM
from werkzeug.exceptions import abort

from vec_search.auth import login_required
from vec_search.db import get_db

#HACK: for now hard codes the location of the config file for AI MODEL
from vec_search.config import AI_MODEL as MODEL

bp = Blueprint('search', __name__)


_MODEL = RobertaForMaskedLM.from_pretrained(MODEL)
_TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)


# we hack the GET & disambiguate a search
@bp.route('/', methods=["GET"])
def index():
    db = get_db()
    if request.method == "GET" and request.args.get("q") is None:
        # TODO: setup pagination
        posts = db.execute(
            'SELECT func_name, path, sha, code, doc FROM post limit 10'
        ).fetchall()
        return render_template('search/index.html', posts=posts)
    elif request.method == "GET":
        # load the LLM to embed the natural lang text to a vec
        q = request.args.get('q')
        # block of code to embed natural language query
        tokens = [_TOKENIZER.cls_token] + _TOKENIZER.tokenize(q) + [_TOKENIZER.eos_token]
        raw_token_ids = tensor(_TOKENIZER.convert_tokens_to_ids(tokens))
        tokens_ids = reshape(raw_token_ids, (1, len(raw_token_ids)))
        context_embeddings = _MODEL(tokens_ids, output_hidden_states=True)
        embeds = context_embeddings.hidden_states[-1].detach().numpy()[0, 0, :]
        embed = embeds.tolist()

        # residual from copy paste
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
              knn_matches.distance as distance
            from knn_matches
            left join post on post.id = knn_matches.rowid
        """
        cur.execute(vec_query, [sqlite_vec.serialize_float32(embed)])
        posts = cur.fetchall()
        cur.close()
        return render_template('search/index.html', posts=posts)

    else:
        raise ValueError("should not ever enter this branch")


# NOTE: this is not fully implemented
@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None

        if not title:
            error = 'Func_name is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'INSERT INTO post (func_name, path, sha)'
                ' VALUES (?, ?, ?)',
                (title, body, g.user['id'])
            )
            db.commit()
            return redirect(url_for('search.index'))

    return render_template('search/create.html')


def get_post(id, check_author=True):
    post = get_db().execute(
        'SELECT p.id, func_name, path, sha'
        ' FROM post p'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    if post is None:
        abort(404, f"Post id {id} doesn't exist.")

    if check_author and post['author_id'] != g.user['id']:
        abort(403)

    return post


# NOTE: this is not fully implemented
@bp.route('/<int:id>/update', methods=('GET', 'POST'))
@login_required
def update(id):
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'UPDATE post SET title = ?, body = ?'
                ' WHERE id = ?',
                (title, body, id)
            )
            db.commit()
            return redirect(url_for('search.index'))

    return render_template('search/update.html', post=post)
