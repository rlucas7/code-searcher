#!/bin/bash

if [[ -z "${OPEN_AI_API_KEY}" ]]; then
  echo "The openai api key env var is not set, please set a value and reinvoke"
  exit 1
fi

# do google authentication to go through the project setup for gemini
gcloud auth application-default login


# login in huggingface hub to get codebert and save token on machine
huggingface-cli login

echo "Enter your programming language, e.g. C, go, java, js, or python:"
read lang
echo "got: $lang"
echo "Enter your initials: "

read initials
echo "got: $initials"
# could give them as cmd line args

# openai key export  and google auth login need to be  performed before start
# modify config.py for lang first

# follow terminal to interact with app
# initdb and for each lang and each user (every time we each run annotate.sh)

flask --app vec_search init-db
flask --app vec_search run --debug

flask --app vec_search export-rad-to-csv rad-$lang-lang-$initials.csv

# with openai
flask --app vec_search gen-llm-rels rad-$lang-lang-$initials.csv llm_gen_rel-openai-$lang-$initials.csv

flask --app vec_search gen-ir-metrics llm_gen_rel-openai-$lang-$initials.csv > metrics-openai-$lang-$initials.txt

# with gemini
flask --app vec_search gen-llm-rels rad-$lang-lang-$initials.csv llm_gen_rel-gemini-$lang-$initials.csv gemini

flask --app vec_search gen-ir-metrics llm_gen_rel-gemini-$lang-$initials.csv > metrics-gemini-$lang-$initials.txt
