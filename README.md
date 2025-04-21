# Book Recommendation System

A comprehensive web application that provides personalized book recommendations based on user queries, combining data from Google Books API with Amazon pricing and review information. This system analyzes books for emotional content, categorizes them, and offers insightful visualizations.

<img width="1434" alt="image" src="https://github.com/user-attachments/assets/b39fc655-de58-4e2f-a5c5-1b092c795236" />

<img width="1428" alt="image" src="https://github.com/user-attachments/assets/d48001e2-0e50-4351-b5ec-28a0061113cb" />

<img width="1432" alt="image" src="https://github.com/user-attachments/assets/15aa48aa-0114-4d35-aafd-574e1e9a317f" />


## Features

- **Natural Language Search**: Find books using everyday language queries
- **Emotional Analysis**: Understand the emotional profile of books
- **Category Classification**: Browse books by fiction/non-fiction categories
- **Amazon Integration**: Get real-time pricing and buying links
- **Review Analysis**: View sentiment analysis and topic extraction from reader reviews
- **Interactive Visualizations**: Explore recommendations through charts and graphs
- **Filters**: Refine recommendations by emotion and category

## System Architecture

The application consists of two main components:

1. **FastAPI Backend** (`main.py`)
   - Provides book recommendations via Google Books API
   - Handles Amazon data scraping for prices and reviews
   - Performs emotional analysis, sentiment analysis, and topic extraction

2. **Streamlit Frontend** (`app.py`)
   - User-friendly interface for searching and exploring recommendations
   - Interactive visualizations of book data
   - Filtering capabilities for refined recommendations

## Requirements

### Backend Requirements
- Python 3.8+
- FastAPI
- Pandas
- Transformers
- LangChain
- Selenium
- BeautifulSoup
- NLTK
- BERTopic
- Sentence-Transformers

### Frontend Requirements
- Streamlit
- Plotly
- Pandas
- Requests
- python-dotenv

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book-recommendation-system.git
   cd book-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Google Books API key:
     ```
     GOOGLE_BOOKS_API_KEY=your_api_key_here
     ```

## Usage

1. Start the backend server:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8002 --reload
   ```

2. Launch the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

3. Navigate to `http://localhost:8501` in your web browser

## API Endpoints

- `/recommendations/` - Get book recommendations based on a query
  ```
  GET /recommendations/?query=science fiction space exploration&max_results=30
  ```

- `/book/amazon-data/{book_id}` - Get Amazon pricing and review data for a specific book
  ```
  GET /book/amazon-data/Gq1kDwAAQBAJ
  ```

## How It Works

1. The user enters a query describing the type of books they're interested in
2. The backend searches for matching books using the Google Books API
3. Books are categorized and analyzed for emotional content
4. When a user requests Amazon data, the backend scrapes pricing and review information
5. Reviews are analyzed for sentiment and key topics
6. All information is presented in an intuitive interface with filtering capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
