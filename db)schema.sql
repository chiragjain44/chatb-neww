-- Minimal table to store documents to be embedded
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT now()
);

-- Example insert
-- INSERT INTO documents (title, content, metadata) VALUES ('Doc 1', 'This is content', '{"source": "import"}');
