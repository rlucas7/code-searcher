#!/bin/bash                                                                     
                                                                                
echo "Enter your programming language, e.g. C, go, java, js, or python:"        
read lang                                                                       
echo "got: $lang"                                                               
                                                                                
echo "Enter your initials: "                                                    
read initials                                                                   
echo "got: $initials"  
# could give them as cmd line args

# openai key export  and google auth login can be included here or performed before start
# modify config.db for lang first too

# follow terminal to interact with app
# initdb and create new userid/pass for each lang and each user (every time we run annotate.sh)

flask --app vec_search init-db 
flask --app vec_search run --debug

flask --app vec_search export-rad-to-csv rad-$lang-lang-$initials.csv 

# with openai
flask --app vec_search gen-llm-rels rad-$lang-lang-$initials.csv llm_gen_rel-openai-$lang-$initials.csv

flask --app vec_search gen-ir-metrics llm_gen_rel-openai-$lang-$initials.csv > metrics-openai-$lang-$initials.txt

# with gemini
flask --app vec_search gen-llm-rels rad-$lang-lang-$initials.csv llm_gen_rel-gemini-$lang-$initials.csv gemini

flask --app vec_search gen-ir-metrics llm_gen_rel-gemini-$lang-$initials.csv > metrics-gemini-$lang-$initials.txt
