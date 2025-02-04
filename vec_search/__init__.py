import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object("vec_search.config")

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route("/hello")
    def hello():
        return "These are not the droids you're looking for"

    from . import db

    db.init_app(app)

    from . import auth

    app.register_blueprint(auth.bp)

    # here we make the search endpoint to be the top level route
    from . import search

    app.register_blueprint(search.bp)
    app.add_url_rule("/", endpoint="index")
    app.add_url_rule("/detail", endpoint="detail")

    return app
