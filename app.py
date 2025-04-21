import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Get API key from environment variables or set a default for development
api_key = os.getenv("GOOGLE_BOOKS_API_KEY", "")


# API endpoints
API_URL = "http://127.0.0.1:8002/recommendations/"
AMAZON_DATA_URL = "http://127.0.0.1:8002/book/amazon-data/"  # New endpoint for Amazon data(scraping)


# Function to count lines in text
def count_lines(text, chars_per_line=80):
    """Estimate number of lines in text based on character count"""
    if not text:
        return 0
    return max(1, len(text) // chars_per_line)


# Function to get book recommendations
def get_recommendations(query, max_results, api_key):
    params = {
        "query": query,
        "max_results": max_results
    }
    if api_key:
        params["api_key"] = api_key
    
    try:
        response = requests.get(API_URL, params=params, timeout=120)
    
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching recommendations: {response.status_code}")
            st.error(response.text)
            return []
    except requests.exceptions.ConnectionError as ce:
        st.error(f"Connection error: {ce}. Make sure your API server is running at {API_URL}")   
        return []
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server is taking too long to process. Try reducing max_results or simplifying your query.")
        return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []


# Function to fetch Amazon data for a specific book
def fetch_amazon_data(book_id):
    """Fetch Amazon data for a specific book on demand"""
    try:
        response = requests.get(f"{AMAZON_DATA_URL}{book_id}", timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching Amazon data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching Amazon data: {str(e)}")
        return None


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Book Recommendation System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            color: #4056A1;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            background: linear-gradient(45deg, #4056A1, #F13C6E);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 5px;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #F13C6E;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }
        .book-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
            border-left: 5px solid #F13C6E;
        }
        .book-title {
            font-size: 1.4rem;
            font-weight: bold;
            color: #4056A1;
            margin-bottom: 5px;
            text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.1);
        }
        .book-author {
            font-style: italic;
            color: #F13C6E;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        .book-rating {
            color: #FF9E00;
            margin-bottom: 10px;
            font-size: 1.1rem;
            font-weight: bold;
        }
        .book-price {
            color: #008000;
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }
        .buy-button {
            background-color: #FF9900;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 15px;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .buy-button:hover {
            background-color: #FF7700;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .amazon-button {
            background-color: #4056A1;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 10px;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 15px;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .amazon-button:hover {
            background-color: #F13C6E;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .review-sentiment-positive {
            background-color: #DEFFDE;
            border-left: 4px solid #00A300;
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .review-sentiment-negative {
            background-color: #FFE8E8;
            border-left: 4px solid #D30000;
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .review-sentiment-neutral {
            background-color: #F0F0F0;
            border-left: 4px solid #888888;
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .review-topics {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 12px;
        }
        .topic-tag {
            background-color: #E5E7F0;
            color: #4056A1;
            border-radius: 15px;
            padding: 4px 10px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .review-summary {
            font-style: italic;
            color: #555;
            background-color: #F9F9F9;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .emotion-bar {
            height: 20px;
            margin: 5px 0;
            border-radius: 5px;
        }
        /* Updated sidebar styles */
        section[data-testid="stSidebar"] {
            background-color: #1E3A8A;
            color: white;
        }
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] h2 {
            color: white !important;
        }
        section[data-testid="stSidebar"] button {
            background-color: #FCD34D !important;
            color: #1E3A8A !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 10px 15px !important;
            margin-top: 20px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        section[data-testid="stSidebar"] button:hover {
            background-color: #F59E0B !important;
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        /* Updated body styles */
        .stApp {
            background: linear-gradient(135deg, #b4c5e4, #8da4d0);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #E5E7EB;
            border-radius: 5px 5px 0px 0px;
            padding: 10px 20px;
            color: #4056A1;
            font-weight: bold;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4056A1 !important;
            color: white !important;
        }
        .view-more-button {
            background-color: #4056A1;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .view-more-button:hover {
            background-color: #F13C6E;
        }
        .emotion-chip {
            display: inline-block;
            padding: 6px 12px;
            margin: 4px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
        }
        .emotion-chip:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        /* Bright section headings */
        h3, h4, strong {
            color: #4056A1 !important;
            font-weight: bold !important;
        }
        /* Enhanced content boxes */
        div.stMarkdown {
            border-radius: 8px;
            padding: 2px;
        }
        /* Plot enhancements */
        div.js-plotly-plot {
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 5px;
            background: white;
        }
        /* Footer enhancement */
        footer {
            border-top: 2px solid #F13C6E;
            padding-top: 10px;
            color: #4056A1 !important;
            font-weight: bold;
        }
        .section-divider {
            margin: 15px 0;
            border-top: 1px dashed #ccc;
        }
        .loading-spinner {
            text-align: center;
            color: #4056A1;
            font-weight: bold;
        }
        .fetch-amazon-link {
            color: #4056A1;
            cursor: pointer;
            text-decoration: underline;
            font-weight: bold;
        }
        .fetch-amazon-link:hover {
            color: #F13C6E;
        }
    </style>
    """, unsafe_allow_html=True)


    # Header
    st.markdown("<h1 class='main-header'>📚 Book Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("""
    Not sure what to read next? Let us help you discover your next favorite book!
    """)


    # Sidebar for input
    st.sidebar.markdown("<h2>Search Settings</h2>", unsafe_allow_html=True)


    query = st.sidebar.text_input("What kind of books are you looking for?",
                                value="classic romance novels",
                                help="Enter keywords or phrases like 'science fiction space travel' or 'self-help productivity'")


    # Fixed max_results 
    max_results = 30


    # Initialize session state for amazon data
    if 'amazon_data' not in st.session_state:
        st.session_state.amazon_data = {}


    # Search button
    if st.sidebar.button("Find Books"):
        with st.spinner("Searching for books matching your query..."):
            st.session_state.recommendations = get_recommendations(query, max_results, api_key)
    
        if not st.session_state.recommendations:
            st.warning("No recommendations found. Try a different query.")
        else:
            st.success(f"Found {len(st.session_state.recommendations)} books that match your query!")
        
            # Update category options after search
            df = pd.DataFrame(st.session_state.recommendations)
            categories = ["Any"] + sorted(df['category'].unique().tolist())
        
            # Add filters to sidebar after search
            st.sidebar.markdown("<h2>Filters</h2>", unsafe_allow_html=True)
            emotion_filter = st.sidebar.selectbox(
                "Dominant Emotion",
                ["Any", "Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"],
                help="Filter books by their dominant emotional tone"
            )
        
            category_filter = st.sidebar.selectbox(
                "Book Category",
                categories,
                help="Filter books by category",
                key="category_filter_updated"
            )


    # Display recommendations if available
    if 'recommendations' in st.session_state and st.session_state.recommendations:
        recommendations = st.session_state.recommendations
    
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(recommendations)
    
        # Get filter values (if they exist in session state)
        emotion_filter = st.session_state.get("emotion_filter", "Any")
        category_filter = st.session_state.get("category_filter_updated", "Any")
    
        # Apply filters
        filtered_df = df.copy()
    
        if emotion_filter != "Any":
            # Find books where the selected emotion has the highest score
            emotion_cols = [f'emotion_{e.lower()}' for e in ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Neutral"]]
            filtered_df['dominant_emotion'] = filtered_df[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
            filtered_df = filtered_df[filtered_df['dominant_emotion'] == emotion_filter.lower()]
    
        if category_filter != "Any":
            filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
        if filtered_df.empty:
            st.warning(f"No books match the selected filters. Try different filter options.")
        else:
            recommendations = filtered_df.to_dict('records')
        
            # Tab layout
            tab1, tab2, tab3 = st.tabs(["Book Cards", "Emotion Analysis", "Category Distribution"])
        
            with tab1:
                st.markdown("<h2 class='sub-header'>Recommended Books</h2>", unsafe_allow_html=True)
            
                # Create a card for each book
                for i, book in enumerate(recommendations):
                    col1, col2 = st.columns([1, 3])
                 
                    # Get the book_id for this book
                    book_id = book['book_id']
                
                    with col1:
                        if book['image']:
                            st.image(book['image'], width=200)
                        else:
                            # Use placeholder image (using a web placeholder instead of local path)
                            st.image("https://via.placeholder.com/200x300?text=No+Cover", width=200)
                
                    with col2:
                        st.markdown(f"<div class='book-title'>{book['title_and_subtitle']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='book-author'>by {book['authors']}</div>", unsafe_allow_html=True)
                    
                        # Rating
                        if book['averageRating'] and str(book['averageRating']).lower() not in ['nan', 'n/a', '']:
                            st.markdown(f"<div class='book-rating'>⭐ {book['averageRating']} ({book['ratingsCount']} ratings)</div>",
                    unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='book-rating'>No ratings available</div>", unsafe_allow_html=True)
                    
                        # Category and publisher
                        st.markdown(f"**Category:** {book['category']} | **Publisher:** {book['publisher']}")
                    
                        # Check if we have Amazon data for this book already
                        amazon_data_loaded = book_id in st.session_state.amazon_data
                     
                        # If Amazon data is not loaded yet, show a button to fetch it
                        if not amazon_data_loaded:
                            fetch_key = f"fetch_amazon_{book_id}"
                            if st.button("🛒 Get Price & Reviews", key=fetch_key, help="Get Amazon price and reviews for this book"):
                                with st.spinner(f"Fetching Amazon data for '{book['title']}'..."):
                                    amazon_data = fetch_amazon_data(book_id)
                                    if amazon_data and "error" not in amazon_data:
                                        st.session_state.amazon_data[book_id] = amazon_data
                                        st.rerun()  # Force a rerun to display the data
                        else:
                            # Amazon data is available, display it
                            amazon_data = st.session_state.amazon_data[book_id]
                         
                            # Display price
                            if 'price' in amazon_data and amazon_data['price'] != "Not available":
                                st.markdown(f"<div class='book-price'>Price: {amazon_data['price']}</div>", unsafe_allow_html=True)
                         
                            # Display buy link
                            if 'buy_link' in amazon_data and amazon_data['buy_link'] != "Not available":
                                st.markdown(f"<a href='{amazon_data['buy_link']}' target='_blank' class='buy-button'>Buy Now</a>", unsafe_allow_html=True)
                     
                        # Book description (truncated to 20 lines)
                        description = book['description']
                        lines_count = count_lines(description)
                        if lines_count > 20:
                            # Show truncated description with "View More" button
                            truncated_desc = ' '.join(description.split()[:150]) + "..."
    
                            # Create a unique key for this book's button and modal
                            modal_key = f"modal_{i}"
                            button_key = f"button_{i}"
    
                            st.markdown(f"**Description:** {truncated_desc}")
    
                            # Create "View More" button that opens a popup
                            if st.button("View Full Description", key=button_key, help="Click to read the full description"):
                                st.session_state[modal_key] = True
    
                            # Create the modal/popup
                            if modal_key in st.session_state and st.session_state[modal_key]:
                                with st.expander("Full Description", expanded=True):
                                    st.write(description)
                                    if st.button("Close", key=f"close_{i}"):
                                        st.session_state[modal_key] = False
                        else:
                            # Show full description for shorter text
                            st.markdown(f"**Description:** {description}")
                    
                        # Emotion distribution 
                        emotions = {
                            'Joy': book['emotion_joy'],
                            'Sadness': book['emotion_sadness'],
                            'Anger': book['emotion_anger'],
                            'Fear': book['emotion_fear'],
                            'Surprise': book['emotion_surprise'],
                            'Neutral': book['emotion_neutral']
                        }
                    
                        # Color mapping for emotions
                        emotion_colors = {
                            'Joy': '#FFD700',      # Gold
                            'Sadness': '#4682B4',  # Steel Blue
                            'Anger': '#DC143C',    # Crimson
                            'Fear': '#800080',     # Purple
                            'Surprise': '#FF8C00', # Dark Orange
                            'Neutral': '#808080'   # Gray
                        }
                    
                        # Display emotional profile as colored chips/pills
                        st.write("**Emotional Profile:**")
                    
                        # Sort emotions by value for better visualization
                        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    
                        html_chips = ""
                        for emotion, score in sorted_emotions:
                            if emotion != 'Disgust':  
                                # Only show emotions with some significance
                                if float(score) > 0.1:
                                    # Scale the opacity based on the score
                                    opacity = max(0.3, float(score))
                                    html_chips += f"""
                                    <div class="emotion-chip" style="background-color: {emotion_colors[emotion]};
                                                                   opacity: {opacity};">
                                        {emotion}: {int(float(score)*100)}%
                                    </div>
                                    """
                    
                        st.markdown(html_chips, unsafe_allow_html=True)
                    
                        
                        if amazon_data_loaded:
                            
                            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                         
                            amazon_data = st.session_state.amazon_data[book_id]
                         
                            # Display Amazon review data (sentiment, topics, summary)
                            if 'review_sentiment' in amazon_data and amazon_data['review_sentiment'] != "neutral" and amazon_data.get('reviews_summary', "") != "No reviews available":
                                sentiment_class = f"review-sentiment-{amazon_data['review_sentiment']}"
                                sentiment_icon = "😃" if amazon_data['review_sentiment'] == "positive" else ("😞" if amazon_data['review_sentiment'] == "negative" else "😐")
                            
                                st.markdown(f"<div class='{sentiment_class}'>{sentiment_icon} <strong>Reader Sentiment:</strong> {amazon_data['review_sentiment'].capitalize()}</div>", unsafe_allow_html=True)
                        
                            # Display review topics
                            if 'review_topics' in amazon_data and amazon_data['review_topics']:
                                st.write("**Reader Mentions:**")
                                topic_html = "<div class='review-topics'>"
                             
                                # Handle both string and list formats for review_topics
                                if isinstance(amazon_data['review_topics'], str):
                                    topics = amazon_data['review_topics'].split(", ")
                                else:
                                    topics = amazon_data['review_topics']
                                 
                                for topic in topics:
                                    if topic:  # Only add if topic is not empty
                                        topic_html += f"<span class='topic-tag'>{topic}</span>"
                                topic_html += "</div>"
                                st.markdown(topic_html, unsafe_allow_html=True)
                        
                            # Display review summary
                            if 'reviews_summary' in amazon_data and amazon_data['reviews_summary'] != "No reviews available":
                                st.write("**Reader Reviews:**")
                             
                                # Handle both string and list formats for reviews
                                if isinstance(amazon_data['reviews_summary'], str):
                                    reviews = amazon_data['reviews_summary'].split(" | ")
                                elif 'reviews' in amazon_data and amazon_data['reviews']:
                                    reviews = amazon_data['reviews']
                                else:
                                    reviews = []
                                 
                                for idx, review in enumerate(reviews[:2]):  # Show max 2 reviews
                                    if review and len(review) > 10:  # Only show substantial reviews
                                        # Truncate very long reviews
                                        if len(review) > 300:
                                            review = review[:300] + "..."
                                        st.markdown(f"<div class='review-summary'>\"{review}\"</div>", unsafe_allow_html=True)
                
                    st.markdown("---")
        
            with tab2:
                st.markdown("<h2 class='sub-header'>Emotional Analysis</h2>", unsafe_allow_html=True)
            
                # Create a radar chart for each book's emotions
                fig = go.Figure()
            
                for i, book in enumerate(recommendations[:5]):  # Limit to first 5 books for clarity
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            book['emotion_joy'],
                            book['emotion_sadness'],
                            book['emotion_anger'],
                            book['emotion_fear'],
                            book['emotion_surprise'],
                            book['emotion_neutral']
                        ],
                        theta=['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Neutral'],
                        fill='toself',
                        name=book['title'][:30] + ('...' if len(book['title']) > 30 else '')
                    ))
            
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    height=600,
                    title="Emotional Profile Comparison of Top 5 Books"
                )
            
                st.plotly_chart(fig, use_container_width=True)
            
                # Average emotion distribution
                st.markdown("<h3>Average Emotional Distribution</h3>", unsafe_allow_html=True)
            
                emotion_cols = [col for col in filtered_df.columns if col.startswith('emotion_')]
                avg_emotions = filtered_df[emotion_cols].mean().reset_index()
                avg_emotions.columns = ['Emotion', 'Score']
                avg_emotions['Emotion'] = avg_emotions['Emotion'].str.replace('emotion_', '')
            
                fig = px.bar(
                    avg_emotions,
                    x='Emotion',
                    y='Score',
                    color='Score',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Average Emotional Distribution in Recommended Books"
                )
            
                st.plotly_chart(fig, use_container_width=True)
        
            with tab3:
                st.markdown("<h2 class='sub-header'>Category Analysis</h2>", unsafe_allow_html=True)
            
                # Category distribution
                category_counts = filtered_df['category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
            
                fig = px.pie(
                    category_counts,
                    values='Count',
                    names='Category',
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    title="Distribution of Book Categories"
                )
            
                st.plotly_chart(fig, use_container_width=True)
            
                # Publication years
                if 'publishedDate' in filtered_df.columns:
                    filtered_df['year'] = pd.to_datetime(filtered_df['publishedDate'], errors='coerce').dt.year
                    year_counts = filtered_df['year'].value_counts().sort_index().reset_index()
                    year_counts.columns = ['Year', 'Count']
                    year_counts = year_counts.dropna()
                
                    if not year_counts.empty:
                        fig = px.bar(
                            year_counts,
                            x='Year',
                            y='Count',
                            title="Publication Years of Recommended Books"
                        )
                    
                        st.plotly_chart(fig, use_container_width=True)
            
                # Publisher analysis
                publisher_counts = filtered_df['publisher'].value_counts().head(10).reset_index()
                publisher_counts.columns = ['Publisher', 'Count']
            
                fig = px.bar(
                    publisher_counts,
                    x='Count',
                    y='Publisher',
                    orientation='h',
                    title="Top Publishers in Recommendations",
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            
                st.plotly_chart(fig, use_container_width=True)


    else:
        # Display placeholder text for first-time visitors
        st.info("Type your interests in the sidebar and tap 'Find Books' to get started!")
    
        # Sample image or placeholder
        st.image("https://via.placeholder.com/800x400?text=Book+Recommendation+System", use_container_width=True)


    # Footer
    st.markdown("---")
    st.markdown("📚 **Book Recommendation System** - Powered by AI")


# Create a Streamlit app function for ASGI compatibility
def create_app():
    return main


# Create the ASGI app instance
app = create_app()


# Run the app directly when this script is executed
if __name__ == "__main__":
    main()