# Amazon Book Scraper Airflow Pipeline

This project contains an Apache Airflow DAG that scrapes Amazon search results for data science books, transforms the data, and loads it into a PostgreSQL database.

## Overview

The DAG performs the following operations:
1. Creates a books table in PostgreSQL if it doesn't exist
2. Creates a metadata table to track scraping progress
3. Scrapes book data from Amazon search results for data science books
4. Transforms and cleans the data
5. Inserts the book data into PostgreSQL
6. Logs scraping metadata

## Prerequisites

- Docker and Docker Compose
- Minimum system requirements:
  - 4GB RAM
  - 2 CPUs
  - 10GB disk space

## Setup Instructions

### 1. Project Structure

Ensure your project has the following structure:
```
airflow-book-scraper/
├── dags/
│   └── fetch_and_store_amazon_books.py
├── logs/
├── plugins/
├── docker-compose.yaml
└── README.md
```

### 2. Configuration

1. Place the DAG code into the `dags/fetch_and_store_amazon_books.py` file
2. Use the provided `docker-compose.yaml` file in the root directory

### 3. Environment Variables

You can customize the Airflow deployment with these environment variables (optional):
```
AIRFLOW_IMAGE_NAME=apache/airflow:2.10.5
AIRFLOW_UID=50000  # Use your user's UID on Linux
_AIRFLOW_WWW_USER_USERNAME=airflow  # Web UI username
_AIRFLOW_WWW_USER_PASSWORD=airflow  # Web UI password
```

### 4. Start the Airflow Services

```bash
# Start Airflow services
docker-compose up -d

# Check if all services are running
docker-compose ps
```

### 5. Access the Airflow Web UI

Open your browser and navigate to:
```
http://localhost:8080
```

Login with the credentials:
- Username: airflow
- Password: airflow

## DAG Details

### DAG ID
`fetch_and_store_amazon_books`

### Schedule
Runs every 20 minutes (`*/20 * * * *`)

### Tasks

1. `create_table`: Creates the books table in PostgreSQL
2. `create_metadata_table`: Creates the metadata table to track scraping progress
3. `fetch_book_data`: Scrapes book data from Amazon 
4. `check_xcom_data`: Verifies the data was successfully scraped
5. `insert_book_data`: Inserts the book data into PostgreSQL
6. `log_metadata`: Logs the current scraping metadata

### Connections

The DAG uses a PostgreSQL connection with the ID `books_connection`. This connection is automatically configured in the provided Docker Compose file.

## Working with the Data

### Database Schema

The books table has the following schema:
```sql
CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    price TEXT,
    rating TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The metadata table has the following schema:
```sql
CREATE TABLE scraping_metadata (
    id SERIAL PRIMARY KEY,
    search_term TEXT NOT NULL,
    last_page_scraped INT DEFAULT 0,
    total_books_scraped INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Accessing the Data

You can access the PostgreSQL database directly at:
```
Host: localhost
Port: 5432
Username: airflow
Password: airflow
Database: airflow
```

Example query to view scraped books:
```sql
SELECT * FROM books ORDER BY created_at DESC LIMIT 10;
```

Example query to view scraping metadata:
```sql
SELECT * FROM scraping_metadata;
```

## Troubleshooting

### Common Issues

1. **Web scraping failures**: Amazon may block scraping attempts. The DAG includes fallback sample data.

2. **Database connection issues**: Verify the PostgreSQL service is running:
   ```bash
   docker-compose ps postgres
   ```

3. **DAG not showing in the UI**: Check the DAG file for syntax errors:
   ```bash
   docker-compose exec airflow-worker python -c "import fetch_and_store_amazon_books"
   ```

### Viewing Logs

Access task logs via the Airflow Web UI or directly:
```bash
docker-compose logs airflow-worker
```

## Customization

### Modifying Search Parameters

To modify the search parameters, update the search query in the `get_amazon_data_books` function:

```python
params = {
    'k': 'your-search-term',  # Change this to search for different books
    'i': 'stripbooks',
    'page': page
}
```

### Adjusting Scraping Frequency

To change how often the DAG runs, modify the `schedule_interval` parameter:

```python
dag = DAG(
    'fetch_and_store_amazon_books',
    default_args=default_args,
    description='A DAG to fetch book data from Amazon and store it in Postgres',
    schedule_interval='0 */6 * * *',  # Change to your desired schedule
    catchup=False,
)
```

## License

This project is open-source and available under the Apache License 2.0.

## Disclaimer

This project is for educational purposes only. Be aware that web scraping may violate Amazon's Terms of Service. Always respect website terms of use and implement proper rate limiting.
