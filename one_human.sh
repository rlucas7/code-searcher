initials=lrr;
for retriever in bm25 codebert; # codet5p
do
  for lang in c java js go python;
  do
    for provider in llama4 aws openai gemini;
    do
      echo "Generating relevances for: $retriever $lang $provider";
      echo "rad-$lang-lang-$initials-$retriever.csv llm_gen_rel-$provider-$lang-$initials-$provider.csv $provider";
      flask --app vec_search gen-llm-rels backups/radzz/rad-$lang-lang-$initials-$retriever.csv backups/human-$initials/llm_gen_rel-$provider-$lang-$retriever-$initials.csv $provider;
      flask --app vec_search gen-ir-metrics backups/human-$initials/llm_gen_rel-$provider-$lang-$retriever-$initials.csv > backups/metrics-$initials/metrics-$provider-$lang-$retriever-$initials.txt;
    done;
  done;
done;