CREATE TABLE IF NOT EXISTS salaries (
    id SERIAL PRIMARY KEY,
    job_title TEXT NOT NULL,
    company_type TEXT,
    city TEXT NOT NULL,
    experience TEXT,
    education TEXT,
    min_salary INTEGER,
    max_salary INTEGER,
    avg_salary REAL,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
