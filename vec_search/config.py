# a config file for the flask app
# these values are used when you run
# `flask --app vec_search`
# if you want something different edit or overwrite this file
# the values are available on the `app.config` object

# TODO: investigate using dynaconf instead, a config tool
# that has a really clean ontology of config and config like items
# e.g. they separate secrets from config-which is nice
import os

DEBUG = True
PREFERRED_URL_SCHEME = 'http'
SECRET_KEY = 'dev'

# NOTE: this assumes you always call from the repo root which is what
# should be happening anyway
DATABASE = f"{os.getcwd()}/var/vec_search-instance/vec_search.sqlite"



# these config values are for the model that is used to generate the embeddings
# for now these need to work with `RobertaTokenizer` and `RobertaForMaskedLM`
# via hf local cache but this will likely change
AI_MODEL = 'microsoft/codebert-base-mlm'

# this file contains the indexed contents of the code repo you have indexed
# I do this via some custom forked code that is in a private repo
# the only things we need in the json entrues are:
# (1) "embeddings" -> vectors for the embeddings
# (2) "sha"        -> commit code for the state of the function when embedded
# (3) "path"       -> relative path from repo root
# (4) "code"       -> a string literal of the function code
# (5) "doc"        -> a string literal of the doc for function
# (6) "func_name"  -> a string literal of the function name, may be of the form
#                       <class-name>.<function-name> if a method on a class.
#
# Some of these entries are used for the browser gui only
# but might be used for other items in future, e.g. link via file:// to the
# local location (on disk) etc.
_JSONL_LOCAL_FILE = "Collections-C.jsonl"

# this corresponds to the sub-directory under space that has the sqlite extension
# development environment
_SQLITE_WORKSPACE = "sqlite-ext" if os.environ.get('USER') == "rlucas" else "ailab"

# for sqlite extensions they are setup as dynamically linked libraries
# that you compile from the extension code via your C-compiler. The path here
# should be absolute and contain the name of the executable file that you
# generate when you setup sqlite-vec
# for more info see the sqlite-vec docs
_SQLITE_VEC_DLL_PATH = f"/Users/{os.environ.get('USER')}/{_SQLITE_WORKSPACE}/sqlite-vec/dist/vec0.dylib"
