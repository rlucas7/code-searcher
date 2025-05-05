"""This module houses the retriever class.
The retriever class encapsulates both semantic and sparse retrievers.
Sparse retrievers leverage the bm25s package.
"""
from typing import Optional, Any

import bm25s
import json
import sqlite_vec

from torch import tensor, reshape, FloatTensor
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM

from vec_search.config import AI_MODEL as MODEL
from vec_search.config import N, DEVICE

from flask import current_app as app

class Retriever:
    def __init__(self, semantic:bool, filepath: Optional[str]=None):
        self.is_semantic = semantic
        if self.is_semantic:
            if MODEL == "Salesforce/codet5p-110m-embedding":
                self._MODEL = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(DEVICE)
                self._TOKENIZER = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            elif MODEL == "microsoft/codebert-base-mlm":
                self._MODEL = RobertaForMaskedLM.from_pretrained(MODEL)
                self._TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)
            else:
                raise ValueError(f"Error: Model config: {MODEL} not currently supported, please open a PR to add...")
        else: # bm25 case is assumed
            if filepath is not None:
                self._set_bm25_retriever(filepath)

    def _set_bm25_retriever(self, filepath:str) -> None:
        """internal method"""
        if self.is_semantic:
            raise ValueError(f"setting a sparse retriever on a semantic model!")
        else:
            #app.logger.info(f"found is_semantic = {self.is_semantic}, initializing bm25 index and retriever...")
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

    def _attach_db(self, db: Any) -> None:
        """attaches the db to the retriever"""
        self.db = db

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
            if MODEL == "microsoft/codebert-base-mlm":
                app.logger.info(f"Running semantic search using codebert on query...")
                # embed natural language query into embedding vector
                tokens = (
                    [self._TOKENIZER.cls_token] + self._TOKENIZER.tokenize(query) + [self._TOKENIZER.eos_token]
                )
                raw_token_ids = tensor(self._TOKENIZER.convert_tokens_to_ids(tokens))
                tokens_ids = reshape(raw_token_ids, (1, len(raw_token_ids)))
                context_embeddings = self._MODEL(tokens_ids, output_hidden_states=True)
                embeds = context_embeddings.hidden_states[-1].detach().numpy()[0, 0, :]
                embed = embeds.tolist()
            elif MODEL == "Salesforce/codet5p-110m-embedding":
                # logic for codet5+
                app.logger.info(f"Running semantic search using codeT5+ to embed query...")
                inputs = self._TOKENIZER.encode(query, return_tensors="pt").to(DEVICE)
                embedding = self._MODEL(inputs)[0]
                embed = embedding.tolist()
            else:
                raise ValueError(f"Error: Model config: {MODEL} not currently supported, please open a PR to add...")
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


class Embedder:
    def __init__(self, hf_model: str = 'microsoft/codebert-base-mlm', device="cpu"):
        self.device = device
        self.hf_model = hf_model
        if hf_model == 'microsoft/codebert-base-mlm':
            self._MODEL = RobertaForMaskedLM.from_pretrained(MODEL)
            self._TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)
        elif hf_model == "Salesforce/codet5p-110m-embedding":
            self._MODEL = AutoModel.from_pretrained(hf_model, trust_remote_code=True).to(device)
            self._TOKENIZER = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        else:
            raise ValueError(f"Error: hf_model = {hf_model} not currently supported, please open a PR to add...")

    def embed(self, result: dict[str, Any])-> list[float]:
        if self.hf_model == 'microsoft/codebert-base-mlm':
            app.logger.log(level=9, msg="Embedding using codebert on result...")
            # this logic aligns w/what I did in the vault fork
            nl_tokens = result['docstring_tokens']
            pl_tokens = result['code_tokens']
            # builds up tensor: "<s> <NL-tokens> </s> <PL-tokens> </s>"
            tokens = [self._TOKENIZER.cls_token]
            tokens.extend(nl_tokens + [self._TOKENIZER.sep_token])
            tokens.extend(pl_tokens + [self._TOKENIZER.eos_token])
            raw_token_ids = self._TOKENIZER.convert_tokens_to_ids(tokens)
            tokens_ids = reshape(tensor(raw_token_ids), (1, len(raw_token_ids)))
            context_embeddings = self._MODEL(tokens_ids, output_hidden_states=True)
            embeds = context_embeddings.hidden_states[-1].detach().numpy()[0, 0, :]
            embed = embeds.tolist()
        elif self.hf_model == "Salesforce/codet5p-110m-embedding":
            app.logger.log(level=9, msg="using codet5+ model ...")
            # here we work directly with the source code text because the tokenizer is somewhat different
            # note that in this next line, inputs is the token ids...
            inputs = self._TOKENIZER.encode(result['original_string'], return_tensors="pt").to(self.device)
            if inputs.shape[1] > self._TOKENIZER.model_max_length:
                app.logger.info(f"result has token length {inputs.shape[1]}...")
                app.logger.info(f"result starts with: {result['original_string'][0:25]}...")
            embedding = self._MODEL(inputs)[0]
            embed = embedding.tolist()
        else:
            raise ValueError(f"Error: hf_model = {self.hf_model} not currently supported in `embed()`, please open a PR to add...")
        return embed