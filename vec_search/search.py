from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from vec_search.auth import login_required
from vec_search.db import get_db

bp = Blueprint('search', __name__)


@bp.route('/')
def index():
    db = get_db()
    # TODO: setup pagination
    posts = db.execute(
        'SELECT func_name, path, sha, code, doc FROM post limit 5'
    ).fetchall()
    print(len(posts))
    for p in posts:
        print(p)
    print(len(posts))
    return render_template('search/index.html', posts=posts)


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
