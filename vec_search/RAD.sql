SELECT
qr.query_id,
qr.post_id,
q.user_id,
GROUP_CONCAT(qr.relevance) AS relevances,
qr.rank,
qr.distance,
q.query,
p.code,
p.doc
FROM query_relevances AS qr, post AS p
INNER JOIN (
  SELECT query_id, query, user_id FROM queries
) AS q ON qr.query_id = q.query_id AND p.id = qr.post_id
GROUP BY 
q.query_id,
user_id,
post_id
;