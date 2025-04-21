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

    class Config:
        orm_mode = True

# Your existing functions
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

def classify_book_categories(df, model_name="facebook/bart-large-mnli", device="cpu"):
    """
    Cleans the categories column and classifies books into predefined categories
    using zero-shot classification on their descriptions.
    
    Args:
        df (pandas.DataFrame): DataFrame containing book data with 'categories' and 'description' columns
        model_name (str): The huggingface model to use for classification
        device (str): Computing device to use ('cpu', 'cuda', 'mps', etc.)
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned categories and new classification
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Clean the categories column - remove brackets and quotes
    result_df['categories'] = result_df['categories'].astype(str).str.replace(r"^\[|\]$", "", regex=True).str.replace("'", "")
    
    # Define candidate categories for classification
    candidate_labels = ["fiction", "non-fiction", "children's fiction", "children's non-fiction"]
    
    # Load the zero-shot classifier with explicit model and device
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    
    # Function to classify each book
    def classify_book(row):
        try:
            description = row['description']
            if not isinstance(description, str) or len(description.strip()) < 10:
                return "unknown"  # Handle empty or very short descriptions
                
            result = classifier(description, candidate_labels)
            # Return the label with the highest score
            return result['labels'][0]
        except Exception as e:
            print(f"Classification error for book id {row.get('book_id', 'unknown')}: {e}")
            return "unknown"
    
    # Apply classification function to the DataFrame
    print(f"Classifying books based on descriptions using model {model_name} on {device}...")
    result_df['category'] = result_df.apply(classify_book, axis=1)
    print(f"Classification complete. Found {result_df['category'].value_counts().to_dict()}")
    
    return result_df

def analyze_book_emotions(df, num_books=None, device="cpu"):
    """
    Analyzes emotions in book descriptions by breaking them into sentences
    and finding the maximum probability for each emotion.
    
    Args:
        df (pandas.DataFrame): DataFrame containing book data with 'book_id' and 'description' columns
        num_books (int, optional): Number of books to analyze. If None, analyzes all books.
        device (str): Device to use for classification ("cpu", "cuda", "mps", etc.)
        
    Returns:
        pandas.DataFrame: Original DataFrame with added emotion score columns
    """
    # Load the emotion classification model
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device
    )
    
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
        
        # Get predictions for each sentence
        predictions = classifier(sentences)
        
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

def book_recommendation_system(query, max_results=30, api_key=None):
    """
    Complete book recommendation system that fetches books and finds relevant ones
    
    Args:
        query (str): The user's query for books
        max_results (int): Maximum number of books to fetch
        api_key (str): Google Books API key
    
    Returns:
        pandas.DataFrame: Dataframe of recommended books
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
    
    # Create embedding database from fetched books
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create documents from descriptions
    documents = []
    for idx, row in processed_df.iterrows():
        doc = Document(
            page_content=f"{row['book_id']} {row['description']}",
            metadata={"book_id": row['book_id']}
        )
        documents.append(doc)
    
    # Create vector database
    db_books = Chroma.from_documents(documents, embedding=embedding_model)
    
    print("Finding semantically similar books")
    
    # Get the most semantically similar books to the original query
    recs = db_books.similarity_search(query, k=min(10, len(processed_df)))
    
    # Get the recommended book IDs
    book_ids = [doc.metadata["book_id"] for doc in recs]
    
    # Return the recommended books
    recommended_books = classify_book_categories(processed_df[processed_df["book_id"].isin(book_ids)],
                                              model_name="facebook/bart-large-mnli", 
                                              device="cpu")
    
    # Only proceed with emotion analysis if there are books to analyze
    if not recommended_books.empty:
        recommendations = analyze_book_emotions(recommended_books)
        return recommendations
    else:
        return pd.DataFrame()

@app.get("/recommendations/", response_model=List[BookRecommendation])
async def get_recommendations(
    query: str = Query(..., description="The search query for book recommendations"),
    max_results: int = Query(30, description="Maximum number of books to fetch"),
    api_key: Optional[str] = Query(None, description="Google Books API key")
):
     # Add logging
    print(f"Received request for query: '{query}' with max_results={max_results}")
    try:
       # Call your book recommendation system
       recommendations_df = book_recommendation_system(query=query, max_results=max_results, api_key=api_key)
    
       if recommendations_df.empty:
           print("no recommendations found") 
           return []
    
       # Convert DataFrame to list of dictionaries
       recommendations_list = recommendations_df.fillna("").to_dict(orient="records")
       print(f"Returning {len(recommendations_list)} recommendations")
       return recommendations_list
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)