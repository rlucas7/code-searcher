# replace language and userid every time annotate workflow runs
# could give them as cmd line args

# openai key export  and google auth login can be included here or performed before start
# modify config.db for lang first too

# follow terminal to interact with app
# initdb and create new userid/pass for each lang and each user (every time we run annotate.sh)

flask --app vec_search init-db 
flask --app vec_search run --debug

flask --app vec_search export-rad-to-csv rad-python-lang-dr.csv 

# with openai
flask --app vec_search gen-llm-rels rad-python-lang-dr.csv llm_gen_rel-openai-python-dr.csv

flask --app vec_search gen-ir-metrics llm_gen_rel-openai-python-dr.csv > metrics-openai-python-dr.txt

# with gemini
flask --app vec_search gen-llm-rels rad-python-lang-dr.csv llm_gen_rel-gemini-python-dr.csv gemini

flask --app vec_search gen-ir-metrics llm_gen_rel-gemini-python-dr.csv > metrics-gemini-python-dr.txt
