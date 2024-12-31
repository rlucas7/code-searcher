DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS vec_items;
DROP TABLE IF EXISTS queries;
DROP TABLE IF EXISTS query_relevances;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE post (
  id INTEGER PRIMARY KEY,
  func_name TEXT NOT NULL,
  path TEXT NOT NULL,
  sha TEXT NOT NULL,
  code TEXT NOT NULL,
  doc TEXT NOT NULL
);

CREATE virtual TABLE vec_items USING vec0(
  rowid INTEGER PRIMARY KEY,
  embedding FLOAT[768] distance_metric=cosine
);

CREATE TABLE queries (
  query_id INTEGER PRIMARY KEY,
  query TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES user (id)
);

CREATE TABLE query_relevances (
  query_id INTEGER NOT NULL,
  post_id INTEGER NOT NULL,
  relevance INTEGER NOT NULL,
  rank INTEGER NOT NULL,
  distance FLOAT NOT NULL
);