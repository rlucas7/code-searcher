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
PREFERRED_URL_SCHEME = "http"
LOG_LEVEL = "DEBUG" # this should always be set to smth
SECRET_KEY = "dev"

# NOTE: this assumes you always call from the repo root which is what
# should be happening anyway
DATABASE = f"{os.getcwd()}/var/vec_search-instance/vec_search.sqlite"


# these config values are for the model that is used to generate the embeddings
# for now these need to work with `RobertaTokenizer` and `RobertaForMaskedLM`
# via hf local cache but this will likely change
# AI_MODEL = "Salesforce/codet5p-110m-embedding" # VEC_DIM 256
AI_MODEL = "microsoft/codebert-base-mlm" # VEC_DIM 768

DEVICE = "cpu"
# the size of the vector embeddings-previously this was a literal in the schema.sql file
# but now we interpolate into that file using this value. The reason is that this makes
# it so that you can configured the embeddings for different models. In other words, you
# will want to set the value for `VEC_DIM` to the dimension of the vectors that are
# returned by `AI_MODEL` chosen above
# VEC_DIM = 256 # for "Salesforce/codet5p-110m-embedding"
VEC_DIM = 768 # for "microsoft/codebert-base-mlm"

# The initial workflow embedded the codebert-base-mlm model into the JSON.
# If you have embeddings already in the JSONL file then keep this false.
# If you want to generate embeddings for a model-perhaps not the model
# used in the jsonl file but used for the retriever, then setting this to
# true will cause the system to feed the code functions through the `AI_MODEL`
# it's handy for experimentations
EMBED_ON_LOAD = False

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
_SQLITE_WORKSPACE = "sqlite-ext" if os.environ.get("USER") == "rlucas" else "ailab"

# for sqlite extensions they are setup as dynamically linked libraries
# that you compile from the extension code via your C-compiler. The path here
# should be absolute and contain the name of the executable file that you
# generate when you setup sqlite-vec
# for more info see the sqlite-vec docs
_SQLITE_VEC_DLL_PATH = (
    f"/Users/{os.environ.get('USER')}/{_SQLITE_WORKSPACE}/sqlite-vec/dist/vec0.dylib"
)

# this config toggles the semantic retriever (true) vs a sparse retriever (false).
# Codet5+ and CodeBERT are tested retrievers which have been tested end-2-end.
SEMANTIC = True

# this value reflects the number of results on the page
N = 10