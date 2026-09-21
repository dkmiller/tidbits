INSERT INTO users_v2 (username, count)
VALUES (?, 1)
ON CONFLICT(username)
DO UPDATE SET count = users_v2.count + 1
RETURNING count;
