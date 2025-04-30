"""This module houses the retriever class.
The retriever class encapsulates both semantic and sparse retrievers.
Sparse retrievers leverage the bm25s package.
"""
from typing import Optional, Any

import bm25s
import json
import sqlite_vec

from torch import tensor, reshape, FloatTensor
from transformers import RobertaTokenizer, RobertaForMaskedLM

from vec_search.config import AI_MODEL as MODEL
from vec_search.config import N

from flask import current_app as app

class Retriever:
    def __init__(self, semantic:bool, db:Any, filepath: Optional[str]=None):
        self.is_semantic = semantic
        self.db = db
        if self.is_semantic:
            self._MODEL = RobertaForMaskedLM.from_pretrained(MODEL)
            self._TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)
        else: # bm25 case is assumed

            if filepath is not None:
                self._set_bm25_retriever(filepath)

    def _set_bm25_retriever(self, filepath:str) -> None:
        """internal method"""
        if self.is_semantic:
            raise ValueError(f"setting a sparse retriever on a semantic model!")
        else:
            app.logger.info(f"found is_semantic = {self.is_semantic}, initializing bm25 index and retriever...")
            with open(filepath, 'r') as jsonl_file:
                result = [json.loads(jline) for jline in jsonl_file.readlines()]
            # following example from
            # https://github.com/xhluca/bm25s/blob/main/examples/index_with_metadata.py
            # we index the source and associate the post.id as the `rowid`
            # then we get the data from the selected top-k entries from `post` table
            # via the `rowid`. The reason we do this is because we want to use the
            # table to populate the SERP
            corpus = [{"text": result[i].get('original_string'), "rowid": str(i)} for i in range(len(result))]
            corpus_text = [doc["text"] for doc in corpus]
            corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")
            self.retriever = bm25s.BM25(corpus=corpus)
            self.retriever.index(corpus_tokens)

    def retrieve(self, user: Any, query:str) -> list[dict[str, Any]]:
        """Retrieve specified number of results based on query and retriever type.

        Args:
            user (Any): A user from the global flask object if the user is logged in, or None if not.
            query (str): A string containing the query.
        """
        if user is not None:
            app.logger.info(f"A `user` value was included with query, storing in `queries` table...")
            # if user is logged in then record the query to query table regardless of semantic or not
            SAVE_QUERY_CMD = "INSERT INTO queries(query, user_id) values (?, ?)"
            self.db.execute(SAVE_QUERY_CMD, [query, user["id"]])
            self.db.commit()
            for row in self.db.execute("SELECT last_insert_rowid()").fetchall():
                query_id = row["last_insert_rowid()"]
        if self.is_semantic:
            app.logger.info(f"Running semantic search on query...")
            # embed natural language query into embedding vector
            tokens = (
                [self._TOKENIZER.cls_token] + self._TOKENIZER.tokenize(query) + [self._TOKENIZER.eos_token]
            )
            raw_token_ids = tensor(self._TOKENIZER.convert_tokens_to_ids(tokens))
            tokens_ids = reshape(raw_token_ids, (1, len(raw_token_ids)))
            context_embeddings = self._MODEL(tokens_ids, output_hidden_states=True)
            embeds = context_embeddings.hidden_states[-1].detach().numpy()[0, 0, :]
            embed = embeds.tolist()
            # now retrieve the top k entries
            cur = self.db.cursor()
            vec_query = f"""
                with knn_matches as (
                  select
                    rowid,
                    distance
                  from vec_items
                  where embedding match ?
                    and k = {N}
                )
                select
                  func_name,
                  path,
                  sha,
                  code,
                  doc,
                  knn_matches.distance as distance,
                  post.id as postid
                from knn_matches
                left join post on post.id = knn_matches.rowid
            """
            # grab the query id to keep the ids client unique
            for row in self.db.execute("SELECT last_insert_rowid()").fetchall():
                query_id = row["last_insert_rowid()"]
            cur.execute(vec_query, [sqlite_vec.serialize_float32(embed)])
            posts = cur.fetchall()
            # now convert all posts into what we want for the client, a bunch of dicts
            for i in range(len(posts)):
                posts[i].update({"search-query": query})
                posts[i].update({"query-id": str(query_id) + f"+post-num-{i}", "rank": str(i)})
            cur.close()
            return posts
        else:
            app.logger.info(f"Running sparse (bm25) search on query...")
            # this is a BM25/sparse retriever
            results, scores = self.retriever.retrieve(bm25s.tokenize(query), k=N)
            # format `doc`s and `score`s to match `posts` from semantic side...
            rows_ids = ','.join([results[0, i].get("rowid") for i in range(N)])
            post_query = f"""SELECT func_name, path, sha, code, doc, post.id as postid FROM post WHERE id IN ({rows_ids})"""
            cur = self.db.cursor()
            cur.execute(post_query)
            posts = cur.fetchall()
            posts.sort(key = lambda x: x.get('postid'))
            rowid_scores = [(int(results[0, i]['rowid']), scores[0, i]) for i in range(N)]
            rowid_scores.sort()
            for i in range(len(posts)):
                posts[i].update({"search-query": query})
                # dividing by 1000 here (mostly removes the effect in the template intended for cosine distance
                posts[i].update({"distance": rowid_scores[i][1]/1000})
            # now rank accoring to score/distance, with bm25 higher is better, whereas for cosine (in semantic True) lower is better
            posts.sort(key=lambda x: -x['distance'])
            for i in range(len(posts)):
                posts[i].update({"query-id": str(query_id) + f"+post-num-{i}", "rank": str(i)})
            return posts