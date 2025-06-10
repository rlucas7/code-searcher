for retriever in bm25 codet5p codebert;
  do
  for lang in c java js go python;
    do
      echo "calculating human agreement on $lang $retriever";
      python human_v_human.py --lang $lang --retriever $retriever;
      echo "";
  done;
done;
