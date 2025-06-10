# C go <- ran these alread but encountered error with java parsing from aws...
# but might want to rerun C with openai, b/c the results are na..,

# also ran these two: java js

# want to rerun: aws/{java,js} AND openai/C
for retriever in codet5p bm25;
  do
  for lang in c java js go python;
  do
    # echo "merging";
    # flask --app vec_search rad-merge rad-$lang-lang-D-$retriever.csv rad-$lang-lang-lrr-$retriever.csv rad-$lang-lang-merged-$retriever.csv
    echo "generating relevances and metrics for: $lang"
    for provider in llama4 aws gemini openai;
    do
      echo "Generating relevances for: $provider";
      flask --app vec_search gen-llm-rels rad-$lang-lang-merged-$retriever.csv llm_gen_rel-$provider-$lang-merged-$retriever.csv $provider
      flask --app vec_search gen-ir-metrics llm_gen_rel-$provider-$lang-merged-$retriever.csv > metrics-$provider-$lang-merged-$retriever.txt
    done;
  done;
done;