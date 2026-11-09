CREATE TABLE IF NOT EXISTS users_v2 (
    username TEXT PRIMARY KEY
    ,count INTEGER
);

CREATE TABLE IF NOT EXISTS users_v3 (
    user_id INTEGER PRIMARY KEY
    ,user_name TEXT
    ,display_name TEXT
);

CREATE TABLE IF NOT EXISTS credentials_v3 (
    id TEXT PRIMARY KEY
    ,user_id INTEGER
    ,public_key BLOB
    ,sign_count INTEGER
    ,transports TEXT -- JSON-encoded
    ,aaguid TEXT
    ,backed_up BOOLEAN
    ,device_type TEXT
    ,nickname TEXT
);

CREATE TABLE IF NOT EXISTS challenges_v3 (
    id TEXT PRIMARY KEY
    ,challenge BYTES
    ,user_id INTEGER
    ,kind TEXT -- "registration" | "authentication"
    ,expires_at INTEGER -- UTC seconds.
);
