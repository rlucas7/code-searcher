DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS vec_items;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE post (
  id INTEGER PRIMARY KEY,
  func_name TEXT NOT NULL,
  path TEXT NOT NULL,
  sha TEXT NOT NULL
);

CREATE virtual TABLE vec_items USING vec0(
  rowid INTEGER PRIMARY KEY,
  embedding FLOAT[768] distance_metric=cosine
);
