# a config file for the flask app
# these values are used when you run
# `flask --app vec_search`
# if you want something different edit or overwrite this file
# the values are available on the `app.config` object

# TODO: investigate using dynaconf instead, a config tool
# that has a really clean ontology of config and config like items
# e.g. they separate secrets from config-which is nice

DEBUG = True
PREFERRED_URL_SCHEME = 'http'
SECRET_KEY = 'dev'
DATABASE = '/Users/rlucas/sqlitevec-w-flask/var/vec_search-instance/vec_search.sqlite'


# these config values are for the model that is used to generate the embeddings
# for now these need to work with `RobertaTokenizer` and `RobertaForMaskedLM`
# via hf local cache but this will likely change
AI_MODEL = 'microsoft/codebert-base-mlm'
