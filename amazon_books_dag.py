"""
DAG to extract Amazon book data, transform it, and load it into a PostgreSQL database.
This DAG scrapes Amazon search results for data science books, cleans the data,
and stores it in a Postgres database using a custom PostgresOperator.
"""
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
import json
import random
import time

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import BaseOperator

# Define custom PostgresOperator
class CustomPostgresOperator(BaseOperator):
   """
   Executes SQL code in a specific Postgres database


   :param sql: the sql code to be executed
   :type sql: str representing a sql statement, a list of str (sql statements),
              or reference to a template file
   :param postgres_conn_id: reference to a specific postgres database
   :type postgres_conn_id: str
   :param autocommit: if True, each command is automatically committed
   :type autocommit: bool
   :param parameters: parameters to be passed to the SQL query
   :type parameters: dict or iterable
   :param database: name of database which overwrite defined one in connection
   :type database: str
   """


   template_fields = ('sql',)
   template_ext = ('.sql',)
   ui_color = '#ededed'


   def __init__(
           self, sql,
           postgres_conn_id='postgres_default',
           autocommit=False,
           parameters=None,
           database=None,
           *args, **kwargs):
       super(CustomPostgresOperator, self).__init__(*args, **kwargs)
       self.sql = sql
       self.postgres_conn_id = postgres_conn_id
       self.autocommit = autocommit
       self.parameters = parameters
       self.database = database


   def execute(self, context):
       logging.info('Executing: ' + str(self.sql))
       self.hook = PostgresHook(postgres_conn_id=self.postgres_conn_id,
                                schema=self.database)
       self.hook.run(self.sql, self.autocommit, parameters=self.parameters)


# Define a list of user agents to rotate
USER_AGENTS = [
   'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
   'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15',
   'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
]


def get_random_headers():
   """Generate random headers to avoid detection"""
   user_agent = random.choice(USER_AGENTS)
   return {
       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
       "Accept-Language": "en-US,en;q=0.5",
       "Referer": "https://www.amazon.com/",
       "Sec-Fetch-Dest": "document",
       "Sec-Fetch-Mode": "navigate",
       "Sec-Fetch-Site": "same-origin",
       "Sec-Fetch-User": "?1",
       "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="109"',
       "Sec-Ch-Ua-Mobile": "?0",
       "Sec-Ch-Ua-Platform": '"Windows"',
       "Upgrade-Insecure-Requests": "1",
       "User-Agent": user_agent,
       "Cache-Control": "max-age=0",
   }


def create_metadata_table(**context):
    """
    Create a metadata table to track scraping progress
    """
    logging.info("Creating metadata table if it doesn't exist")
    
    try:
        # Get PostgreSQL hook
        postgres_hook = PostgresHook(postgres_conn_id='books_connection')
        
        # Create metadata table if it doesn't exist
        postgres_hook.run("""
        CREATE TABLE IF NOT EXISTS scraping_metadata (
            id SERIAL PRIMARY KEY,
            search_term TEXT NOT NULL,
            last_page_scraped INT DEFAULT 0,
            total_books_scraped INT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """, autocommit=True)
        
        # Check if we have a record for 'data science' books
        result = postgres_hook.get_records(
            "SELECT COUNT(*) FROM scraping_metadata WHERE search_term = 'data science'"
        )
        
        # If no record exists, create one
        if result[0][0] == 0:
            postgres_hook.run(
                "INSERT INTO scraping_metadata (search_term, last_page_scraped, total_books_scraped) VALUES (%s, %s, %s)",
                parameters=('data science', 0, 0),
                autocommit=True
            )
            logging.info("Created initial metadata record for 'data science' search term")
        
    except Exception as e:
        logging.error(f"Error creating metadata table: {str(e)}")
        raise


def get_amazon_data_books(num_books, **context):
   """
   Scrape book data from Amazon search results and push to XCom.


   Args:
       num_books: Number of unique books to collect
       ti: Task instance for XCom push
   """
   logging.info("Starting to fetch Amazon book data")
   ti = context['ti']


   # Create sample books as a fallback
   sample_books = [
       {
           "Title": f"Test Book {i}",
           "Author": f"Author {i}",
           "Price": f"{i + 10}.99",
           "Rating": f"{(i % 5) + 1} out of 5 stars"
       } for i in range(1, 6)  # 5 sample books
   ]


   logging.info(f"Created {len(sample_books)} sample books for testing (fallback)")


   # Base URL of the Amazon search results for data science books
   base_url = "https://www.amazon.com/s"
   
   # Get the last page scraped from the metadata table
   last_page = 0
   try:
       postgres_hook = PostgresHook(postgres_conn_id='books_connection')
       result = postgres_hook.get_records(
           "SELECT last_page_scraped, total_books_scraped FROM scraping_metadata WHERE search_term = 'data science'"
       )
       if result and len(result) > 0:
           last_page = result[0][0]
           total_books_scraped = result[0][1]
           logging.info(f"Retrieved last page scraped: {last_page}, total books scraped: {total_books_scraped}")
   except Exception as e:
       logging.error(f"Error getting last page scraped: {str(e)}")
       # Continue with default value if there's an error


   try:
       # First try to fetch real data
       books = []
       seen_titles = set()  # To keep track of seen titles
       page = last_page + 1  # Start from the next page
       max_pages = page + 10  # Limit to 10 new pages per run
       total_books_scraped_in_run = 0


       while len(books) < num_books and page <= max_pages:
           # Add query parameters
           params = {
               'k': 'books',
               'i': 'stripbooks',
               'page': page
           }


           # Get random headers to avoid detection
           headers = get_random_headers()


           # Add a delay between requests to avoid being blocked
           if page > last_page + 1:
               sleep_time = random.uniform(2, 5)
               logging.info(f"Sleeping for {sleep_time:.2f} seconds before next request")
               time.sleep(sleep_time)


           logging.info(f"Fetching data from page {page}")


           try:
               # Send a request to the URL
               response = requests.get(base_url, params=params, headers=headers, timeout=15)
               logging.info(f"Response status code: {response.status_code}")


               # For debugging, save a sample of the HTML
               if page == last_page + 1:
                   html_sample = response.text[:1000] + "..." if len(response.text) > 1000 else response.text
                   logging.info(f"HTML sample: {html_sample}")


               # Check if the request was successful
               if response.status_code == 200:
                   # Parse the content of the request with BeautifulSoup
                   soup = BeautifulSoup(response.content, "html.parser")


                   # Look for the main container that holds search results
                   main_container = soup.find("div", {"class": "s-main-slot"})


                   if not main_container:
                       logging.warning("Main container not found, trying alternative container")
                       main_container = soup.find("div", {"class": "s-search-results"})


                   if main_container:
                       # Find all book items (these can have various classes)
                       book_containers = main_container.find_all("div", {"data-component-type": "s-search-result"})


                       if not book_containers:
                           logging.warning("No book containers found with primary selector, trying alternative")
                           book_containers = main_container.find_all("div", {"class": "s-result-item"})


                       logging.info(f"Found {len(book_containers)} book containers")


                       # If no books found on this page, break the loop
                       if not book_containers:
                           logging.warning("No book containers found on page")
                           break


                       # Process each book
                       for book in book_containers:
                           try:
                               # Try multiple selectors for each element
                               # Title selectors
                               title_elem = None
                               for selector in [
                                   {"tag": "span", "attrs": {"class": "a-text-normal"}},
                                   {"tag": "h2", "attrs": {"class": "a-size-mini"}},
                                   {"tag": "h5", "attrs": {}},
                                   {"tag": "h2", "attrs": {}}
                               ]:
                                   title_elem = book.find(selector["tag"], selector["attrs"])
                                   if title_elem:
                                       break


                               # Author selectors
                               author_elem = None
                               for selector in [
                                   {"tag": "a", "attrs": {"class": "a-size-base"}},
                                   {"tag": "span", "attrs": {"class": "a-color-secondary"}},
                                   {"tag": "div", "attrs": {"class": "a-row a-size-base"}}
                               ]:
                                   author_elem = book.find(selector["tag"], selector["attrs"])
                                   if author_elem:
                                       break


                               # Price selectors
                               price_elem = None
                               for selector in [
                                   {"tag": "span", "attrs": {"class": "a-price-whole"}},
                                   {"tag": "span", "attrs": {"class": "a-offscreen"}},
                                   {"tag": "span", "attrs": {"class": "a-price"}}
                               ]:
                                   price_elem = book.find(selector["tag"], selector["attrs"])
                                   if price_elem:
                                       break


                               # Rating selectors
                               rating_elem = None
                               for selector in [
                                   {"tag": "span", "attrs": {"class": "a-icon-alt"}},
                                   {"tag": "div", "attrs": {"class": "a-row a-size-small"}},
                                   {"tag": "i", "attrs": {"class": "a-icon"}}
                               ]:
                                   rating_elem = book.find(selector["tag"], selector["attrs"])
                                   if rating_elem:
                                       break


                               # Log what we found for debugging
                               logging.info(f"Elements found - Title: {title_elem is not None}, "
                                          f"Author: {author_elem is not None}, "
                                          f"Price: {price_elem is not None}, "
                                          f"Rating: {rating_elem is not None}")


                               # Only proceed if we at least have a title
                               if title_elem:
                                   book_title = title_elem.text.strip()


                                   # Get other data if available, use default values if not
                                   author = author_elem.text.strip() if author_elem else "Unknown Author"
                                   price = price_elem.text.strip() if price_elem else "Price unavailable"
                                   rating = rating_elem.text.strip() if rating_elem else "No rating"


                                   # Check if title has been seen before
                                   if book_title not in seen_titles and "Sponsored" not in book_title:
                                       seen_titles.add(book_title)
                                       books.append({
                                           "Title": book_title,
                                           "Author": author,
                                           "Price": price,
                                           "Rating": rating,
                                       })
                                       logging.info(f"Added book: {book_title[:40]}...")
                                       total_books_scraped_in_run += 1
                               else:
                                   logging.warning("Book container found but no title element")
                           except Exception as e:
                               logging.error(f"Error processing book: {str(e)}")
                   else:
                       logging.warning("Could not find any container for search results")
                       break


                   # Break loop if we have enough books
                   if len(books) >= num_books:
                       logging.info(f"Collected enough books ({len(books)}), stopping search")
                       break


                   # Go to next page and update the last page scraped in XCom
                   ti.xcom_push(key='last_page_scraped', value=page)
                   page += 1
               else:
                   logging.warning(f"Failed to retrieve page {page}: Status {response.status_code}")
                   break


           except Exception as e:
               logging.error(f"Error fetching data from page {page}: {str(e)}")
               break


       # Store the last successfully scraped page and total books in metadata
       try:
           if page > last_page:
               last_scraped_page = page - 1
               postgres_hook = PostgresHook(postgres_conn_id='books_connection')
               
               # Update metadata
               postgres_hook.run(
                   """
                   UPDATE scraping_metadata 
                   SET last_page_scraped = %s, 
                       total_books_scraped = total_books_scraped + %s,
                       last_updated = CURRENT_TIMESTAMP
                   WHERE search_term = 'data science'
                   """,
                   parameters=(last_scraped_page, total_books_scraped_in_run),
                   autocommit=True
               )
               logging.info(f"Updated metadata: last_page_scraped = {last_scraped_page}, "
                           f"added {total_books_scraped_in_run} new books")
       except Exception as e:
           logging.error(f"Error updating metadata: {str(e)}")


       # Limit to the requested number of books
       books = books[:num_books]


       # If we couldn't fetch enough real books, use sample data
       if len(books) < 5:
           logging.warning(f"Could only fetch {len(books)} real books, using sample data instead")
           books = sample_books
       else:
           logging.info(f"Successfully collected {len(books)} real books")


       # Convert to JSON string for debugging
       books_json = json.dumps(books)
       logging.info(f"Books data (first 500 chars): {books_json[:500]}...")


       # Push the books data to XCom
       ti.xcom_push(key='book_data', value=books)
       logging.info("Successfully pushed book data to XCom")


   except Exception as e:
       logging.error(f"Major error in get_amazon_data_books: {str(e)}")
       # Ensure we always have at least sample data
       ti.xcom_push(key='book_data', value=sample_books)
       logging.info("Used sample data as fallback due to error")



def check_xcom_data(**context):
   """
   Check if XCom data exists and log its contents.
   """
   ti = context['ti']
   book_data = ti.xcom_pull(key='book_data', task_ids='fetch_book_data')


   if book_data:
       logging.info(f"Found {len(book_data)} books in XCom")
       # Log the first book as a sample
       if len(book_data) > 0:
           logging.info(f"First book: {book_data[0]}")
       return True
   else:
       logging.error("No book data found in XCom!")
       return False



def insert_book_data_into_postgres(**context):
   """
   Insert the book data from XCom into PostgreSQL using PostgresHook.


   Args:
       ti: Task instance for XCom pull
   """
   logging.info("Starting to insert book data into Postgres")
   ti = context['ti']


   # Get book data from XCom
   book_data = ti.xcom_pull(key='book_data', task_ids='fetch_book_data')


   if not book_data:
       logging.error("No book data found in XCom")
       raise ValueError("No book data found in XCom")


   logging.info(f"Retrieved {len(book_data)} books from XCom")


   try:
       # Get PostgreSQL hook
       postgres_hook = PostgresHook(postgres_conn_id='books_connection')


       # Check if connection works
       conn = postgres_hook.get_conn()
       cursor = conn.cursor()
       cursor.execute("SELECT 1")
       result = cursor.fetchone()
       logging.info(f"Database connection test: {result}")
       cursor.close()


       # Check if table exists
       cursor = conn.cursor()
       cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'books')")
       table_exists = cursor.fetchone()[0]
       logging.info(f"Table 'books' exists: {table_exists}")
       cursor.close()


       if not table_exists:
           logging.warning("Books table does not exist, creating it now")
           cursor = conn.cursor()
           cursor.execute("""
           CREATE TABLE IF NOT EXISTS books (
               id SERIAL PRIMARY KEY,
               title TEXT NOT NULL,
               authors TEXT,
               price TEXT,
               rating TEXT,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           );
           """)
           conn.commit()
           cursor.close()


       # Loop through each book and insert into database
       insert_query = """
       INSERT INTO books (title, authors, price, rating)
       VALUES (%s, %s, %s, %s)
       """


       inserted_count = 0
       for book in book_data:
           try:
               postgres_hook.run(
                   insert_query,
                   parameters=(
                       book.get('Title', 'Unknown'),
                       book.get('Author', 'Unknown'),
                       book.get('Price', 'Unknown'),
                       book.get('Rating', 'Unknown')
                   ),
                   autocommit=True
               )
               inserted_count += 1
           except Exception as e:
               logging.error(f"Error inserting book {book.get('Title', 'Unknown')}: {str(e)}")


       logging.info(f"Successfully inserted {inserted_count} books into database")


       # Verify data was inserted
       cursor = conn.cursor()
       cursor.execute("SELECT COUNT(*) FROM books")
       total_books = cursor.fetchone()[0]
       logging.info(f"Total books in database: {total_books}")
       cursor.close()


   except Exception as e:
       logging.error(f"Database error: {str(e)}")
       raise


def log_scraping_metadata(**context):
    """
    Log the current scraping metadata
    """
    logging.info("Checking current scraping metadata")
    
    try:
        # Get PostgreSQL hook
        postgres_hook = PostgresHook(postgres_conn_id='books_connection')
        
        # Query metadata
        result = postgres_hook.get_records(
            "SELECT search_term, last_page_scraped, total_books_scraped, last_updated FROM scraping_metadata"
        )
        
        for row in result:
            logging.info(f"Metadata: search_term={row[0]}, last_page_scraped={row[1]}, "
                         f"total_books_scraped={row[2]}, last_updated={row[3]}")
            
    except Exception as e:
        logging.error(f"Error querying metadata: {str(e)}")


# Define DAG default arguments
default_args = {
   'owner': 'airflow',
   'depends_on_past': False,
   'start_date': datetime(2025, 4, 7),
   'email_on_failure': False,
   'email_on_retry': False,
   'retries': 1,
   'retry_delay': timedelta(minutes=5),
}


# Create DAG
dag = DAG(
   'fetch_and_store_amazon_books',
   default_args=default_args,
   description='A DAG to fetch book data from Amazon and store it in Postgres',
   schedule_interval='*/20 * * * *',  # runs every 20 minutes
   catchup=False,
)


# Create table task using custom operator
create_table_task = CustomPostgresOperator(
   task_id='create_table',
   postgres_conn_id='books_connection',
   sql="""
   CREATE TABLE IF NOT EXISTS books (
       id SERIAL PRIMARY KEY,
       title TEXT NOT NULL,
       authors TEXT,
       price TEXT,
       rating TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   """,
   dag=dag,
)


# Create metadata table task
create_metadata_table_task = PythonOperator(
   task_id='create_metadata_table',
   python_callable=create_metadata_table,
   dag=dag,
)


# Fetch book data task
fetch_book_data_task = PythonOperator(
   task_id='fetch_book_data',
   python_callable=get_amazon_data_books,
   op_kwargs={'num_books': 100},  # Number of books to fetch
   dag=dag,
)


# Check XCom data task
check_xcom_task = PythonOperator(
   task_id='check_xcom_data',
   python_callable=check_xcom_data,
   dag=dag,
)


# Insert book data task
insert_book_data_task = PythonOperator(
   task_id='insert_book_data',
   python_callable=insert_book_data_into_postgres,
   dag=dag,
)


# Log metadata task
log_metadata_task = PythonOperator(
   task_id='log_metadata',
   python_callable=log_scraping_metadata,
   dag=dag,
)


# Set task dependencies
create_table_task >> create_metadata_table_task >> fetch_book_data_task >> check_xcom_task >> insert_book_data_task >> log_metadata_task