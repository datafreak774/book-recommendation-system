#main.py file
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import uuid
from typing import List, Optional
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Global storage for recent recommendations
stored_recommendations = []

# Import your model initialization functions
from models import init_zero_shot_classifier, init_emotion_classifier, init_embedding_model, init_sentiment_analyzer
 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

print("⚠️ IMPORTANT: Make sure to run this notebook from top to bottom sequentially!")
print("Models are loaded only once and reused to conserve memory.")

sentiment_analyzer = init_sentiment_analyzer(device=0)  # Use appropriate device


print("Loading zero-shot classification model...")
zero_shot_classifier = pipeline("zero-shot-classification",
                               model="facebook/bart-large-mnli",
                               device=0)  # Use appropriate device (0 for first GPU, "cpu" for CPU)


print("Loading emotion classification model...")
emotion_classifier = pipeline(
   "text-classification",
   model="j-hartmann/emotion-english-distilroberta-base",
   top_k=None,
   device=0  # Use appropriate device
)


# Initialize embedding model if needed
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


print("All models loaded successfully")


app = FastAPI(title="Book Recommendation API")


# Enable CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)


# Pydantic model for book response
class BookRecommendation(BaseModel):
  book_id: str
  title: str
  title_and_subtitle: str
  authors: str
  description: str
  publisher: Optional[str]
  publishedDate: Optional[str]
  pageCount: Optional[int]
  categories: Optional[str]
  averageRating: Optional[float]
  ratingsCount: Optional[int]
  image: Optional[str]
  category: str
  emotion_anger: float
  emotion_disgust: float
  emotion_fear: float
  emotion_joy: float
  emotion_sadness: float
  emotion_surprise: float
  emotion_neutral: float
  #Without updating this model, FastAPI won't include these new fields in the response even though they exist in your dataframe. FastAPI uses this model to validate and filter the data before sending it to the client.
  price: Optional[str]
  buy_link: Optional[str]
  review_sentiment: Optional[str]
  review_sentiment_score: Optional[float]
  review_topics: Optional[str]
  reviews_summary: Optional[str]
  reviews: Optional[str]


  class Config:
      orm_mode = True
     
# Startup event to load models once when the app starts
@app.on_event("startup")
async def startup_event():
   # Initialize models and store them in app.state
   # This way they're loaded only once when the app starts
   app.state.zero_shot_classifier = init_zero_shot_classifier(device=0)
   app.state.emotion_classifier = init_emotion_classifier(device=0)
   app.state.embedding_model = init_embedding_model()
   print("All models loaded successfully and ready for use")

    
# Download NLTK resources for sentiment analysis
nltk.download('vader_lexicon', quiet=True)


def extract_reviews_from_page(driver):
   """
   Extract reviews from the current page using the HTML structure from the screenshots
   Args:
      driver: Selenium WebDriver instance
    
   Returns:
      list: List of review texts
   """
   reviews = []
   try:
      # Get page source for BeautifulSoup parsing
      page_source = driver.page_source
      soup = BeautifulSoup(page_source, 'html.parser')
    
      print("Extracting reviews using the exact structure from inspection...")
    
      # Look for divs with data-hook="review" or "review-collapsed"
      review_elements = soup.select('div[data-hook="review"], div[data-hook="review-collapsed"]')
      print(f"Found {len(review_elements)} review elements")
    
      if not review_elements:
          # Try an alternative selector based on the HTML inspection screenshot
          review_elements = soup.select('div[id^="customer_review-"]')
          print(f"Second attempt found {len(review_elements)} review elements")
    
      for review_elem in review_elements:
          try:
              # First look for the review-text-content element
              review_text_elem = review_elem.select_one('span[data-hook="review-body"] span')
            
              if not review_text_elem:
                  # Alternative selectors based on the HTML structure
                  review_text_elem = review_elem.select_one('div.a-expander-content.review-text-content span')
            
              if not review_text_elem:
                  # Try another alternative from the screenshots
                  review_text_elem = review_elem.select_one('div.a-expander-content.review-text-content')
            
              if review_text_elem:
                  # Get clean text
                  review_text = review_text_elem.get_text().strip()
                  if review_text and len(review_text) > 10:
                      reviews.append(review_text)
                      print(f"Found review: {review_text[:50]}...")
          except Exception as e:
              print(f"Error extracting individual review: {e}")
              continue
    
      # If still no reviews, try another approach based directly on the HTML inspection
      if not reviews:
          print("Attempting another extraction approach...")
        
          # Look for the specific classes shown in your first screenshot
          for span_elem in soup.select('div.a-expander-content.review-text-content.a-expander-partial-collapse-content span'):
              text = span_elem.get_text().strip()
              if text and len(text) > 10 and text not in reviews:
                  reviews.append(text)
                  print(f"Found review with alternate approach: {text[:50]}...")
    
      
      if not reviews:
          print("Attempting extraction with very specific selectors...")
        
          
          selectors = [
              'div.a-expander-content.review-text-content.a-expander-partial-collapse-content span',
              'div.a-expander-content.reviewText.review-text-content span',
              'div[data-hook="review-collapsed"] span',
              'div.review-text-content span'
          ]
        
          for selector in selectors:
              for elem in soup.select(selector):
                  text = elem.get_text().strip()
                  if text and len(text) > 10 and text not in reviews:
                      reviews.append(text)
                      print(f"Found review with specific selector: {text[:50]}...")
            
              if reviews:
                  break
                
   except Exception as e:
      print(f"Error in review extraction: {e}")
   return reviews


def scrape_amazon_data(book_title, author):
   """
   Scrape Amazon data for a specific book
   Args:
      book_title (str): Title of the book
      author (str): Author of the book
    
   Returns:
      dict: Dictionary containing price, buy link, and reviews
   """
   # Setup Chrome options
   chrome_options = Options()
   chrome_options.add_argument("--headless")  # Run in headless mode
   chrome_options.add_argument("--no-sandbox")
   chrome_options.add_argument("--disable-dev-shm-usage")
   chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
   chrome_options.add_argument("--window-size=1920,1080")  # Set larger window size
   result = {
      "price": "Not available",
      "buy_link": "Not available",
      "reviews": []
   }
   try:
      # Initialize WebDriver
      driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
      # Search for the book on Amazon
      search_query = f"{book_title} {author} book"
      search_url = f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"
    
      driver.get(search_url)
      print(f"Searching for: {search_query}")
      time.sleep(3)  # Wait for page to load
    
      # Find the first book result
      try:
          # Look for book links
          book_link_selectors = [
              "a.a-link-normal.s-underline-text",
              ".s-result-item h2 a.a-link-normal",
              ".s-result-item .a-link-normal.s-no-outline",
              ".a-link-normal.s-text-normal"
          ]
        
          book_link = None
          book_url = None
        
          for selector in book_link_selectors:
              try:
                  print(f"Trying selector: {selector}")
                  elements = driver.find_elements(By.CSS_SELECTOR, selector)
                  for element in elements:
                      href = element.get_attribute("href")
                      # Check if it's likely a book link
                      if href and (("/dp/" in href) or ("-ebook/" in href)):
                          # Remove any anchors from the URL
                          book_url = href.split("#")[0]
                          book_link = element
                          break
                  if book_link:
                      break
              except Exception as e:
                  print(f"Error with selector {selector}: {e}")
                  continue
        
          if book_url:
              print(f"Found book link: {book_url}")
              result["buy_link"] = book_url
            
              # Navigate directly to the book page
              driver.get(book_url)
              print("Navigated to book page")
              time.sleep(4)  # Wait for page to load
            
              # Get book price - try multiple price selectors
              price_selectors = [
                  ".a-price .a-offscreen",
                  "#kindle-price",
                  ".kindle-price #kindle-price",
                  ".a-color-price",
                  ".a-size-medium.a-color-price"
              ]
            
              for price_selector in price_selectors:
                  try:
                      print(f"Trying price selector: {price_selector}")
                      price_elements = driver.find_elements(By.CSS_SELECTOR, price_selector)
                      for price_element in price_elements:
                          price_text = price_element.get_attribute("textContent") or price_element.text
                          if price_text and '$' in price_text:
                              result["price"] = price_text.strip()
                              print(f"Found price: {result['price']}")
                              break
                      if result["price"] != "Not available":
                          break
                  except Exception as e:
                      print(f"Error with price selector {price_selector}: {e}")
                      continue
            
              # Try to navigate to the Reviews tab
              try:
                  # First try clicking the Reviews tab if available
                  try:
                      reviews_tab = WebDriverWait(driver, 3).until(
                          EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#customerReviews'], .a-link-normal[href*='#review']"))
                      )
                      reviews_tab.click()
                      print("Clicked reviews tab")
                      time.sleep(2)
                  except:
                      print("Couldn't click reviews tab, continuing with current page")
                
                  # Extract reviews from current page
                  current_page_reviews = extract_reviews_from_page(driver)
                  result["reviews"].extend(current_page_reviews)
                
                  # If we didn't find reviews, try navigating to dedicated reviews page
                  if not result["reviews"]:
                      # Try different selectors for reviews links
                      review_link_selectors = [
                          "a[data-hook='see-all-reviews-link-foot']",
                          "a.a-link-emphasis[href*='customer-reviews']",
                          "#reviews-medley-footer a.a-link-emphasis"
                      ]
                    
                      for selector in review_link_selectors:
                          try:
                              review_links = driver.find_elements(By.CSS_SELECTOR, selector)
                              if review_links:
                                  review_url = review_links[0].get_attribute("href")
                                  print(f"Found review page link: {review_url}")
                                
                                  # Navigate to reviews page
                                  driver.get(review_url)
                                  print("Navigated to reviews page")
                                  time.sleep(3)
                                
                                  # Extract reviews from dedicated page
                                  dedicated_page_reviews = extract_reviews_from_page(driver)
                                  result["reviews"].extend(dedicated_page_reviews)
                                  break
                          except:
                              continue
                
                  print(f"Total reviews extracted: {len(result['reviews'])}")
                
              except Exception as e:
                  print(f"Error processing reviews: {e}")
          else:
              print("Couldn't find book link")
            
      except Exception as e:
          print(f"Error finding book: {e}")
    
      return result
   except Exception as e:
      print(f"Error scraping Amazon data: {e}")
      return result
   finally:
      try:
          driver.quit()
      except:
          pass



def analyze_reviews(reviews):
    """
    Perform sentiment analysis and topic modeling on reviews using transformer models
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        dict: Dictionary with sentiment scores and topics
    """
    if not reviews:
        return {
            "sentiment_score": 0,
            "sentiment": "neutral",
            "common_topics": []
        }
    
    import numpy as np
    
    # Combine all reviews for topic modeling
    all_text = " ".join(reviews)
    
    # Analyze sentiment for each review using the pre-loaded model
    sentiments = []
    
    for review in reviews:
        # Skip empty reviews or those that are too long
        if not review.strip() or len(review) > 512:
            continue
            
        try:
            # Get sentiment prediction
            result = sentiment_analyzer(review)[0]
            
            # Map sentiment labels and scores to our format
            if result['label'] == 'POSITIVE':
                score = result['score']  # Already between 0 and 1
            else:  # NEGATIVE
                score = -result['score']  # Make negative for consistency
                
            sentiments.append(score)
        except Exception:
            # Handle any exceptions with the model
            continue
    
    # Calculate average sentiment score
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    
    # Determine sentiment label (maintaining same thresholds)
    if avg_sentiment >= 0.05:
        sentiment_label = "positive"
    elif avg_sentiment <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    # Topic modeling with BERTopic
    common_topics = []
    
    # Define improved fallback topic extraction function
    def fallback_topics():
        import re
        from collections import Counter
        import string
        
        # Create a simple stopwords list
        stopwords = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what", 
                     "when", "where", "how", "who", "which", "this", "that", "these", "those",
                     "then", "just", "so", "than", "such", "both", "through", "about", "for",
                     "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                     "having", "do", "does", "did", "doing", "would", "should", "could",
                     "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
                     "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd",
                     "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
                     "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                     "doesn't", "don't", "didn't", "very", "really", "quite", "that", "book",
                     "read", "reading", "reader", "books", "chapter", "page", "pages"}
        
        # Normalize text
        text = all_text.lower()
        
        # Remove punctuation and split into words
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # Extract words with at least 4 letters, not in stopwords
        words = [word for word in text.split() if len(word) >= 4 and word not in stopwords]
        
        # Get word frequency
        word_counts = Counter(words)
        
        # Get most common meaningful words
        common = [word for word, count in word_counts.most_common(10) if count > 1]
        
        return common[:5]  # Return top 5 words
    
    # Initialize BERTopic model locally if we have enough reviews
    bertopic_succeeded = False
    if len(reviews) >= 2:
        try:
            print("Loading BERTopic model within analyze_reviews function...")
            
            # Import necessary libraries
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            from sentence_transformers import SentenceTransformer
            import nltk
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            # Get English stopwords
            from nltk.corpus import stopwords
            stop_words = list(stopwords.words('english'))
            
            # Add additional common words that are often not helpful for topic modeling
            additional_stops = [
                "book", "read", "author", "story", "character", "characters", "novel",
                "really", "lot", "way", "make", "just", "like", "well", "there", "like", "works",
                "time", "even", "much", "many", "also", "very", "thing", "things", "quite",
                "actually", "say", "said", "one", "two", "three", "first", "second", "third",
                "new", "know", "get", "go", "see", "use", "used", "using", "think", "thought"
            ]
            
            stop_words.extend(additional_stops)
            
            # Configure CountVectorizer with stopwords and n-gram range
            vectorizer = CountVectorizer(
                stop_words=stop_words,
                ngram_range=(1, 2),  # Allow single words and bigrams
                min_df=1,  # Lower min_df for small datasets
                max_df=0.9  # Maximum document frequency (appear in at most 90% of documents)
            )
            
            # Initialize BERTopic with customized parameters
            topic_model = BERTopic(
                embedding_model="all-MiniLM-L6-v2",  # Small model for efficiency
                vectorizer_model=vectorizer,
                n_gram_range=(1, 2),
                min_topic_size=2,  # smaller for better results with few reviews
                nr_topics="auto",  # Automatically determine number of topics
                calculate_probabilities=False,  # Faster processing
                verbose=True
            )
            
            print("Attempting to use BERTopic for topic modeling...")
            
            # Clean reviews - remove empty ones and limit length
            cleaned_reviews = [r for r in reviews if r.strip() and len(r) <= 2048]
            
            # Fit model and transform documents
            topics, _ = topic_model.fit_transform(cleaned_reviews)
            
            # Get topic information
            topic_info = topic_model.get_topic_info()
            
            # Extract topics (excluding -1 which is noise)
            valid_topic_ids = [t for t in set(topics) if t != -1]
            
            if valid_topic_ids:
                # Process each valid topic
                for topic_id in valid_topic_ids[:3]:  # Limit to top 3 topics
                    # Get top words with weights for this topic
                    topic_words = topic_model.get_topic(topic_id)
                    
                    # Only include words with weight above threshold - lower threshold
                    for word, weight in topic_words[:5]:  # Check top 5 words per topic
                        if weight > 0.003 and word not in common_topics:
                            common_topics.append(word)
                
                # Limit to top 5 topics
                common_topics = common_topics[:5]
                
                if common_topics:
                    print(f"BERTopic succeeded. Topics: {common_topics}")
                    bertopic_succeeded = True
            
        except Exception as e:
            print(f"BERTopic failed with error: {str(e)}")
            bertopic_succeeded = False
    
    # Use fallback if BERTopic failed or wasn't available
    if not bertopic_succeeded or not common_topics:
        print("Using fallback topic extraction method")
        common_topics = fallback_topics()
    
    # Round sentiment score to match original format
    avg_sentiment = round(float(avg_sentiment), 2)
    
    return {
        "sentiment_score": avg_sentiment,
        "sentiment": sentiment_label,
        "common_topics": common_topics
    }



def scrape_books_to_dataframe(books_list):
  """
  Scrape data for multiple books and return as DataFrame
   Args:
      books_list: List of dictionaries with book_title and author keys
    
  Returns:
      DataFrame: Contains book information
  """
  results = []
  for book_info in books_list:
      title = book_info['book_title']
      author = book_info['author']
    
      print(f"\nScraping data for '{title}' by {author}...")
      book_data = scrape_amazon_data(title, author)
    
      # Merge all reviews into a single text
      all_reviews = "\n\n".join(book_data["reviews"]) if book_data["reviews"] else "No reviews found"
    
      # Create a row for the DataFrame
      book_row = {
          "book_title": title,
          "author": author,
          "buy_link": book_data["buy_link"],
          "price": book_data["price"],
          "reviews": all_reviews
      }
    
      results.append(book_row)
      print(f"Completed scraping for '{title}'")
   # Create DataFrame from results
  df = pd.DataFrame(results)
  return df



def fetch_books_data(query, max_results=30, api_key=None):
  """
  Fetches books from Google Books API and prepares them for vector similarity search.

  Parameters:
      query (str): The search term
      max_results (int): Max number of books to fetch
      api_key (str): Google Books API key

  Returns:
      descriptions (list of str)
      metadata (list of dict)
      df (pandas.DataFrame): DataFrame of all book metadata
  """
  url = "https://www.googleapis.com/books/v1/volumes"
  params = {
      'q': query,
      'maxResults': max_results,
  }

  if api_key:
      params['key'] = api_key

  response = requests.get(url, params=params)
  descriptions = []
  metadata = []

  if response.status_code == 200:
      data = response.json()
      books = data.get('items', [])

      for book in books:
          info = book.get('volumeInfo', {})
          title = info.get('title', 'N/A')
          subtitle = info.get('subtitle', 'N/A')
          full_title = f"{title}: {subtitle}" if subtitle else title
          authors = ", ".join(info.get('authors', ['N/A']))
          description = info.get('description', '').strip()
          if not description:
              description = "No description available"
          descriptions.append(description)
          book_info = {
              'title': full_title,
              'subtitle':subtitle,
              'authors': authors,
              'publisher': info.get('publisher', 'N/A'),
              'publishedDate': info.get('publishedDate', 'N/A'),
              'pageCount': info.get('pageCount', 'N/A'),
              'categories': info.get('categories', []),
              'averageRating': info.get('averageRating', 'N/A'),
              'ratingsCount': info.get('ratingsCount', 'N/A'),
              'image': info.get('imageLinks', {}).get('thumbnail', ''),
              'description': description
          }
          metadata.append(book_info)
      # Create a DataFrame from the metadata
      df = pd.DataFrame(metadata)
  else:
      print(f"❌ Error {response.status_code}: {response.text}")
      df = pd.DataFrame()  # empty df on failure
  return df



def process_book_data(df):
  """
  Process book data to clean and transform it according to requirements.
   Args:
      df (pandas.DataFrame): The original book dataframe
    
  Returns:
      pandas.DataFrame: The processed book dataframe
  """
  # Make a copy to avoid modifying the original
  df_processed = df.copy()
   # Convert to datetime and handle errors
  df_processed['publishedDate'] = pd.to_datetime(df_processed['publishedDate'], errors='coerce')
   # Extract year
  df_processed['publishedYear'] = df_processed['publishedDate'].dt.year
   # Calculate age of the book (as of 2025)
  df_processed['bookAge'] = 2025 - df_processed['publishedYear']
   # Count words in description
  df_processed["words_in_description"] = df_processed["description"].str.split().str.len()
   # Filter books with at least 15 words in description
  book_missing_25_words = df_processed[df_processed["words_in_description"] >= 25]
   # Extract main title from title
  book_missing_25_words['main_title'] = book_missing_25_words['title'].str.split(':').str[0].str.strip()
   # Create title and subtitle combined field
  book_missing_25_words["title_and_subtitle"] = (
      np.where(book_missing_25_words["subtitle"].isna(),
              book_missing_25_words["title"],
              book_missing_25_words[["main_title", "subtitle"]].astype(str).agg(": ".join, axis=1))
  )
   # Generate UUID for each book
  book_missing_25_words['book_id'] = [str(uuid.uuid4()) for _ in range(len(book_missing_25_words))]
   # Define new column order
  new_order = [
      'book_id', 'main_title', 'subtitle', 'title_and_subtitle', 'authors',
      'description', 'words_in_description', 'publisher', 'publishedDate',
      'bookAge', 'pageCount', 'categories', 'averageRating', 'ratingsCount', 'image'
  ]
   # Apply the new order (make sure all these columns exist)
  book_missing_25_words = book_missing_25_words[new_order]
   # Create tagged description
  book_missing_25_words["tagged_description"] = book_missing_25_words[["book_id", "description"]].astype(str).agg(" ".join, axis=1)
   # Drop specified columns
  book_missing_25_words = book_missing_25_words.drop(["subtitle", "bookAge", "words_in_description"], axis=1)
   # Rename columns
  book_missing_25_words.rename(columns={'main_title': 'title'}, inplace=True)
  return book_missing_25_words


def classify_book_categories(df, app):
   """
   Cleans the categories column and classifies books into predefined categories
   using zero-shot classification on their descriptions.
  
   Args:
       df (pandas.DataFrame): DataFrame containing book data
       app: The FastAPI app instance to access models
      
   Returns:
       pandas.DataFrame: DataFrame with cleaned categories and new classification
   """
   # Make a copy to avoid modifying the original
   result_df = df.copy()
  
   # Clean the categories column - remove brackets and quotes
   result_df['categories'] = result_df['categories'].astype(str).str.replace(r"^\[|\]$", "", regex=True).str.replace("'", "")
  
   # Define candidate labels for classification
   candidate_labels = ["fiction", "non-fiction", "children's fiction", "children's non-fiction"]
  
   # Function to classify each book
   def classify_book(row):
       try:
           description = row['description']
           if not isinstance(description, str) or len(description.strip()) < 10:
               return "unknown"  # Handle empty or very short descriptions
              
           # Use the classifier from app state
           result = app.state.zero_shot_classifier(description, candidate_labels)
           # Return the label with the highest score
           return result['labels'][0]
       except Exception as e:
           print(f"Classification error for book id {row.get('book_id', 'unknown')}: {e}")
           return "unknown"
  
   # Apply classification function to the DataFrame
   print(f"Classifying books based on descriptions...")
   result_df['category'] = result_df.apply(classify_book, axis=1)
   print(f"Classification complete. Found {result_df['category'].value_counts().to_dict()}")
  
   return result_df


def analyze_book_emotions(df, app, num_books=None):
   """
   Analyzes emotions in book descriptions by breaking them into sentences
   and finding the maximum probability for each emotion.
  
   Args:
       df (pandas.DataFrame): DataFrame containing book data
       app: The FastAPI app instance to access models
       num_books (int, optional): Number of books to analyze
      
   Returns:
       pandas.DataFrame: Original DataFrame with added emotion score columns
   """
   # Define emotion labels
   emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
  
   # Initialize lists to store results
   book_id = []
   emotion_scores = {label: [] for label in emotion_labels}
  
   # Helper function to calculate max emotion scores
   def calculate_max_emotion_scores(predictions):
       per_emotion_scores = {label: [] for label in emotion_labels}
      
       for prediction in predictions:
           # Don't sort, but find each emotion by label name
           for label in emotion_labels:
               # Find the score for this specific label
               score = next((item["score"] for item in prediction if item["label"] == label), 0)
               per_emotion_scores[label].append(score)
              
       return {label: np.max(scores) for label, scores in per_emotion_scores.items()}
  
   # Determine how many books to process
   if num_books is None:
       num_books = len(df)
   else:
       num_books = min(num_books, len(df))
  
   print(f"Analyzing emotions in {num_books} book descriptions...")
  
   # Process each book description
   for i in range(num_books):
       # Get book ID and description
       current_book_id = df["book_id"].iloc[i]
       book_id.append(current_book_id)
      
       # Split description into sentences
       description = df["description"].iloc[i]
       sentences = description.split(".")
      
       # Get predictions for each sentence using model from app state
       predictions = app.state.emotion_classifier(sentences)
      
       # Calculate maximum scores for each emotion
       max_scores = calculate_max_emotion_scores(predictions)
      
       # Store results
       for label in emotion_labels:
           emotion_scores[label].append(max_scores[label])
  
   print("Emotion analysis complete!")
  
   # Create result DataFrame
   result_df = pd.DataFrame({"book_id": book_id})
  
   # Add emotion scores
   for label in emotion_labels:
       result_df[f"emotion_{label}"] = emotion_scores[label]
  
   # Join result with original dataframe
   df_with_emotions = pd.merge(df, result_df, on="book_id", how="left")
  
   return df_with_emotions


def book_recommendation_system_without_amazon(query, app, max_results=30, api_key=None):
   """
   Complete book recommendation system that fetches books and finds relevant ones
   without Amazon data integration
  
   Args:
       query (str): The user's query for books
       app: The FastAPI app instance to access models
       max_results (int): Maximum number of books to fetch
       api_key (str): Google Books API key
      
   Returns:
       pandas.DataFrame: Dataframe of recommended books with Amazon data
   """
   print(f"Fetching books for query: '{query}'")
  
   # Fetch books based on the query
   df = fetch_books_data(query, max_results, api_key)
  
   if df.empty:
       print("No books found from the API")
       return pd.DataFrame()
   print(f"Processing {len(df)} books")
  
   # Process the books
   processed_df = process_book_data(df)
  
   if processed_df.empty:
       print("No books remain after processing")
       return pd.DataFrame()
   print("Creating embedding database")
  
   # Create documents from descriptions
   documents = []
   for idx, row in processed_df.iterrows():
       doc = Document(
           page_content=f"{row['book_id']} {row['description']}",
           metadata={"book_id": row['book_id']}
       )
       documents.append(doc)
  
   # Create vector database using model from app state
   db_books = Chroma.from_documents(documents, embedding=app.state.embedding_model)
  
   print("Finding semantically similar books")
  
   # Get the most semantically similar books to the original query
   recs = db_books.similarity_search(query, k=min(10, len(processed_df)))
  
   # Get the recommended book IDs
   book_ids = [doc.metadata["book_id"] for doc in recs]
  
   # Return the recommended books
   recommended_books = classify_book_categories(
       processed_df[processed_df["book_id"].isin(book_ids)],
       app
   )
  
   # Only proceed with emotion analysis if there are books to analyze
   if not recommended_books.empty:
       recommendations = analyze_book_emotions(recommended_books, app)
       return recommendations
   else:
       return pd.DataFrame()


def integrate_amazon_data_with_recommendations(recommendations_df):
  """
  Takes a dataframe of book recommendations and adds Amazon data for each book
   Args:
      recommendations_df (pandas.DataFrame): DataFrame containing book recommendations with title and authors
    
  Returns:
      pandas.DataFrame: Enhanced DataFrame with Amazon data
  """
  # Create a list of dictionaries with book_title and author for each book
  books_to_scrape = []
  for idx, row in recommendations_df.iterrows():
      title = row['title']
      # Handle authors being a list or string
      authors = row['authors']
      if isinstance(authors, list):
          authors = authors[0] if authors else "Unknown"
    
      books_to_scrape.append({
          "book_title": title,
          "author": authors
      })
   # Use the existing function to scrape Amazon data
  amazon_df = scrape_books_to_dataframe(books_to_scrape)
   # Process the Amazon data to extract sentiment analysis for each book
  amazon_enhanced_df = amazon_df.copy()
  amazon_enhanced_df['review_sentiment'] = ""
  amazon_enhanced_df['review_sentiment_score'] = 0.0
  amazon_enhanced_df['review_topics'] = ""
  amazon_enhanced_df['reviews_summary'] = ""
  for idx, row in amazon_enhanced_df.iterrows():
      # Split the combined reviews text back into a list
      reviews_list = row['reviews'].split("\n\n") if row['reviews'] != "No reviews found" else []
    
      # Analyze reviews if they exist
      if reviews_list:
          review_analysis = analyze_reviews(reviews_list)
          amazon_enhanced_df.at[idx, 'review_sentiment'] = review_analysis['sentiment']
          amazon_enhanced_df.at[idx, 'review_sentiment_score'] = review_analysis['sentiment_score']
          amazon_enhanced_df.at[idx, 'review_topics'] = ", ".join(review_analysis['common_topics'])
          # Store a summary of the first 3 reviews (or fewer if there aren't 3)
          amazon_enhanced_df.at[idx, 'reviews_summary'] = " | ".join(reviews_list[:3]) if reviews_list else "No reviews available"
   # Rename columns for merging
  amazon_enhanced_df = amazon_enhanced_df.rename(columns={
      'book_title': 'title',
      'author': 'authors'
  })
   # Merge the original recommendations with the Amazon data
  # The left_index and right_index ensure we maintain the original order
  merged_df = recommendations_df.merge(
      amazon_enhanced_df[['title', 'authors', 'price', 'buy_link',
                         'review_sentiment', 'review_sentiment_score',
                         'review_topics', 'reviews_summary', 'reviews']],
      on=['title', 'authors'],
      how='left'
  )
   # Fill NaN values with appropriate defaults
  merged_df['price'] = merged_df['price'].fillna("Not available")
  merged_df['buy_link'] = merged_df['buy_link'].fillna("Not available")
  merged_df['review_sentiment'] = merged_df['review_sentiment'].fillna("neutral")
  merged_df['review_sentiment_score'] = merged_df['review_sentiment_score'].fillna(0.0)
  merged_df['review_topics'] = merged_df['review_topics'].fillna("")
  merged_df['reviews_summary'] = merged_df['reviews_summary'].fillna("No reviews available")
  merged_df['reviews'] = merged_df['reviews'].fillna("No reviews found")
  return merged_df


# Update the endpoint to use the app state models
@app.get("/recommendations/", response_model=List[BookRecommendation])
async def get_recommendations(
   query: str = Query(..., description="The search query for book recommendations"),
   max_results: int = Query(30, description="Maximum number of books to fetch"),
   api_key: Optional[str] = Query(None, description="Google Books API key")
):
   print(f"Received request for query: '{query}' with max_results={max_results}")
   try:
       # Call your book recommendation system with app instance passed
       recommendations_df = book_recommendation_system_without_amazon(
           query=query,
           app=app,  # Pass the app to access models
           max_results=max_results,
           api_key=api_key
       )
      
       if recommendations_df.empty:
           print("no recommendations found")
           return []
      
       # Convert data types before converting to dict
       # Convert timestamps to strings
       if 'publishedDate' in recommendations_df.columns:
           recommendations_df['publishedDate'] = recommendations_df['publishedDate'].astype(str)
      
       # Convert ratings to proper numeric types
       if 'averageRating' in recommendations_df.columns:
           recommendations_df['averageRating'] = pd.to_numeric(
               recommendations_df['averageRating'].replace('N/A', np.nan),
               errors='coerce'
           )
          
       if 'ratingsCount' in recommendations_df.columns:
           recommendations_df['ratingsCount'] = pd.to_numeric(
               recommendations_df['ratingsCount'].replace('N/A', np.nan),
               errors='coerce'
           ).astype('Int64')  
          
       if 'pageCount' in recommendations_df.columns:
           recommendations_df['pageCount'] = pd.to_numeric(
               recommendations_df['pageCount'].replace('N/A', np.nan),errors='coerce').astype('Int64')  # Use nullable integer type
          
       # Add placeholder fields for Amazon data that will be filled later
       recommendations_df['price'] = "Click to fetch"
       recommendations_df['buy_link'] = None
       recommendations_df['review_sentiment'] = None
       recommendations_df['review_sentiment_score'] = None
       recommendations_df['review_topics'] = None
       recommendations_df['reviews_summary'] = None
       recommendations_df['reviews'] = None   
      
       # Replace NaN values with None for JSON serialization, then convert to dict
       recommendations_df = recommendations_df.replace({np.nan: None})
       recommendations_list = recommendations_df.to_dict(orient="records")
      
       # Store these recommendations in global storage for later lookup
       global stored_recommendations
       stored_recommendations = recommendations_list
      
       print(f"Returning {len(recommendations_list)} recommendations")
       return recommendations_list
   except Exception as e:
       print(f"Error processing request: {str(e)}")
       raise
if __name__ == "__main__":
  uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
 
 
 
@app.get("/book/amazon-data/{book_id}")
async def get_amazon_data(book_id: str):
   """
   Fetches Amazon data (price, buy link, reviews) for a specific book.
  
   Args:
       book_id (str): The ID of the book to fetch Amazon data for
      
   Returns:
       dict: Amazon data including price, buy link, and reviews
   """
   try:
       
       book_title = None
       author = None
       for book in stored_recommendations:
           if book.get('book_id') == book_id:
               book_title = book.get('title')
               author = book.get('authors')
               break
      
       if not book_title or not author:
           return {"error": "Book not found"}
      
       # Now scrape Amazon data for this specific book
       amazon_data = scrape_amazon_data(book_title, author)
      
       # Process the reviews for sentiment analysis
       reviews_list = amazon_data.get('reviews', [])
       review_analysis = analyze_reviews(reviews_list)
      
       # Create response with Amazon data and analysis
       response = {
           "book_id": book_id,
           "price": amazon_data.get('price', 'Not available'),
           "buy_link": amazon_data.get('buy_link', 'Not available'),
           "review_sentiment": review_analysis.get('sentiment', 'neutral'),
           "review_sentiment_score": review_analysis.get('sentiment_score', 0.0),
           "review_topics": review_analysis.get('common_topics', []),
           "reviews_summary": " | ".join(reviews_list[:3]) if reviews_list else "No reviews available",
           "reviews": reviews_list
       }
      
       return response
   except Exception as e:
       print(f"Error fetching Amazon data: {str(e)}")
       return {"error": str(e)}
