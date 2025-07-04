"""
Bulk Sentiment Analysis Service
This application provides sentiment analysis for text data through a Flask web service.
It supports both single text analysis and bulk processing of files.
"""

import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot

from flask import Flask, request, Response, make_response
from transformers import pipeline, AutoConfig
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import seaborn as sns
import time
import requests
from requests.auth import HTTPBasicAuth
import logging
import warnings
import sys
import gc

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.utils._pytree.*')
warnings.filterwarnings('ignore', message='.*resume_download.*')

# Configure logging to be less verbose
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Disable Flask's default logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load environment variables
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Initialize sentiment analysis pipeline lazily
pipe = None

def get_sentiment_pipeline():
    global pipe
    if pipe is None:
        print("Initializing sentiment analysis model...")
        # Use smaller model configuration
        config = AutoConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        config.torchscript = True  # Enable TorchScript for better memory usage
        
        pipe = pipeline("text-classification", 
                       model="distilbert-base-uncased-finetuned-sst-2-english",
                       config=config,
                       framework="pt",
                       device="cpu")  # Explicitly use CPU
    return pipe

# Welcome messages and help information
WELCOME_MESSAGE = """👋 Welcome to the Sentiment Analysis Bot!

To get started, you can:
1. Type 'start' to begin
2. Type 'help' for instructions
3. Type 'hi' or 'hello' for a greeting
4. Send any text to analyze its sentiment
5. Upload a file for bulk analysis"""

HELP_MESSAGE = """🤖 Sentiment Analysis Bot Help

Commands:
- start: Begin using the bot
- help: Show this help message
- hi/hello: Get a greeting

Analysis Options:
1. Single Text Analysis:
   - Simply send any text message

2. Bulk File Analysis:
   - Upload a file (Excel, CSV, or TXT)
   - File should have a column with reviews
   - Supported formats: .xlsx, .csv, .txt

Example:
- Send: "I love this product!"
- Result: Sentiment analysis with emoji

Need more help? Just ask!"""

START_MESSAGE = """🚀 Let's analyze some sentiments!

You can:
1. Send any text message for instant analysis
2. Upload a file for bulk processing

Ready when you are!"""

def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def rating_to_sentiment(rating):
    try:
        rating = float(rating)
        if pd.isna(rating):
            return None
        if rating >= 4:
            return "POSITIVE"
        elif rating <= 2:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    except (ValueError, TypeError):
        print(f"Invalid rating value: {rating}")
        return None

def analyze_text(text: str) -> str:
    """
    Analyze the sentiment of a given text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Sentiment label ('POSITIVE', 'NEGATIVE', or 'NEUTRAL')
    """
    try:
        # Input validation
        if not isinstance(text, str):
            logger.warning(f"Invalid input type: {type(text)}. Converting to string.")
            text = str(text)
        
        if not text or text.strip() == '':
            logger.info("Empty text received, returning NEUTRAL")
            return "NEUTRAL"
            
        # Clean text
        text = text.strip()
        if len(text) > 512:
            logger.warning(f"Text truncated from {len(text)} to 512 characters")
            text = text[:512]
            
        # Get pipeline and analyze sentiment
        sentiment_pipe = get_sentiment_pipeline()
        result = sentiment_pipe(text)
        
        # Force garbage collection after analysis
        gc.collect()
        
        if result and isinstance(result, list) and 'label' in result[0]:
            label = result[0]["label"].upper()
            score = result[0]["score"]
            
            # Handle uncertain predictions
            if 0.4 < score < 0.6:
                logger.info(f"Uncertain prediction (score: {score}), returning NEUTRAL")
                return "NEUTRAL"
                
            logger.info(f"Sentiment: {label} (confidence: {score:.2f})")
            return label
            
        logger.warning("Invalid model output format, returning NEUTRAL")
        return "NEUTRAL"
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return "NEUTRAL"

def process_file(file_path):
    file_ext = file_path.split('.')[-1].lower()

    try:
        if file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame(lines, columns=['review'])
        else:
            print(f"Unsupported file format: {file_ext}")
            return None, None

        print("Columns in file:", df.columns.tolist())

        text_column = None
        rating_column = None
        possible_text_columns = ['review text', 'review', 'text', 'comment', 'feedback', 'content', 'message', 'body']
        possible_rating_columns = ['rating', 'score', 'ratings']

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in possible_text_columns:
                text_column = col
            if col_lower in possible_rating_columns:
                rating_column = col

        if text_column is None and not df.empty:
            text_column = df.columns[0]
            print(f"No standard text column found, defaulting to: {text_column}")

        print(f"Selected text column: {text_column}")
        print(f"Selected rating column: {rating_column}")

        if text_column:
            df[text_column] = df[text_column].astype(str).str.strip().replace('nan', '')
            print("Sample text values:", df[text_column].head().tolist())

            if rating_column and df[rating_column].notna().any():
                print("Using Rating column for sentiment")
                df['sentiment'] = df[rating_column].apply(rating_to_sentiment)
                invalid_count = df['sentiment'].isna().sum()
                if invalid_count > 0:
                    print(f"Found {invalid_count} invalid ratings, using model for those")
                    df.loc[df['sentiment'].isna(), 'sentiment'] = df.loc[df['sentiment'].isna(), text_column].apply(analyze_text)
            else:
                print("No valid Rating column, using model for sentiment")
                df['sentiment'] = df[text_column].apply(analyze_text)

            print("Raw sentiment results:", df['sentiment'].value_counts().to_dict())

            label_mapping = {"POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL"}
            df['sentiment'] = df['sentiment'].map(label_mapping).fillna("NEUTRAL")
            print("Updated sentiment results:", df['sentiment'].value_counts().to_dict())

            sentiment_counts = df['sentiment'].value_counts().to_dict()
            total = len(df)

            stats = {
                'positive': sentiment_counts.get('POSITIVE', 0),
                'negative': sentiment_counts.get('NEGATIVE', 0),
                'neutral': sentiment_counts.get('NEUTRAL', 0),
                'total': total
            }

            return df, stats
        else:
            print("No text column identified")
            return None, None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None

def create_visualization(stats):
    """
    Create visualization of sentiment analysis results.
    
    Args:
        stats (dict): Dictionary containing sentiment statistics
        
    Returns:
        str: Path to the saved visualization file
    """
    try:
        stats = {key: 0 if pd.isna(value) else value for key, value in stats.items()}

        labels = ['Positive', 'Negative', 'Neutral']
        values = [stats['positive'], stats['negative'], stats['neutral']]

        # Create figure with a specific size
        fig = plt.figure(figsize=(10, 6))
        
        # Create pie chart
        plt.subplot(1, 2, 1)
        plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#99ff99', '#ff9999', '#66b3ff'], startangle=90)
        plt.title('Sentiment Distribution')

        # Create bar chart
        plt.subplot(1, 2, 2)
        sns.barplot(x=labels, y=values)
        plt.title('Sentiment Counts')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')

        # Adjust layout and save
        plt.tight_layout()
        
        # Generate unique filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        filename = f'sentiment_analysis_{timestamp}.png'
        filepath = os.path.join('static', filename)
        
        # Save and close the figure
        plt.savefig(filepath)
        plt.close(fig)
        
        return filepath
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint that returns a greeting message"""
    resp = MessagingResponse()
    resp.message(WELCOME_MESSAGE)
    return str(resp)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze endpoint for processing text and files"""
    try:
        # Initialize Twilio response
        resp = MessagingResponse()
        
        # Get incoming message
        incoming_msg = request.form.get('Body', '').strip()
        media_url = request.form.get('MediaUrl0', None)
        
        # Log incoming request details
        logger.info(f"Received request - Message: {incoming_msg[:50]}... Media: {'Yes' if media_url else 'No'}")

        # Handle special keywords
        if incoming_msg.lower() in ['hi', 'hello']:
            resp.message(WELCOME_MESSAGE)
            return str(resp)
        
        elif incoming_msg.lower() == 'help':
            resp.message(HELP_MESSAGE)
            return str(resp)
        
        elif incoming_msg.lower() == 'start':
            resp.message(START_MESSAGE)
            return str(resp)

        # Handle file uploaded via Twilio
        if media_url:
            try:
                logger.info(f"Processing media from URL: {media_url}")
                response = requests.get(
                    media_url,
                    auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                )
                response.raise_for_status()
                
                # Create unique filename
                timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
                filename = f"twilio_file_{timestamp}"
                
                # Determine file extension from content type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'excel' in content_type or 'spreadsheet' in content_type:
                    file_ext = '.xlsx'
                elif 'csv' in content_type:
                    file_ext = '.csv'
                else:
                    file_ext = '.txt'  # Default to text
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + file_ext)
                
                # Save file
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Process file
                results_df, stats = process_file(file_path)
                
                if results_df is not None and stats is not None:
                    # Create visualization
                    viz_path = create_visualization(stats)
                    
                    # Prepare response message
                    summary = (f"📊 Sentiment Analysis Results 📊\n\n"
                             f"Total reviews analyzed: {stats['total']}\n"
                             f"✅ Positive: {stats['positive']} ({stats['positive']/stats['total']*100:.1f}%)\n"
                             f"❌ Negative: {stats['negative']} ({stats['negative']/stats['total']*100:.1f}%)\n"
                             f"➖ Neutral: {stats['neutral']} ({stats['neutral']/stats['total']*100:.1f}%)")
                    
                    # Send text response
                    msg = resp.message(summary)
                    
                    # Attach visualization if created
                    if viz_path:
                        msg.media(request.host_url.rstrip('/') + '/' + viz_path.replace('\\', '/'))
                    
                    return str(resp)
                else:
                    resp.message("Could not process the file. Please ensure it contains a column with reviews and is in Excel, CSV or TXT format.")
                    return str(resp)
                
            except Exception as e:
                logger.error(f"Error processing media: {str(e)}")
                resp.message(f"Sorry, I couldn't process your file: {str(e)}")
                return str(resp)

        # Handle text input
        elif incoming_msg:
            if len(incoming_msg.strip()) == 0:
                resp.message("Please send some text to analyze or type 'help' for instructions.")
                return str(resp)
            
            # Analyze sentiment
            result = analyze_text(incoming_msg)
            emoji = "✅" if result == "POSITIVE" else "❌" if result == "NEGATIVE" else "➖"
            resp.message(f"Sentiment: {emoji} {result}")
            return str(resp)

        # Nothing provided
        else:
            resp.message("Please send a message or upload a file to analyze sentiments. Type 'help' for instructions.")
            return str(resp)
            
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        # Return error response in Twilio format
        resp = MessagingResponse()
        resp.message("Sorry, something went wrong. Please try again later.")
        return str(resp)

@app.route('/version')
def version():
    """Endpoint to check Python version and environment details"""
    return {
        'python_version': sys.version,
        'platform': sys.platform,
        'environment': os.environ.get('RENDER_ENVIRONMENT', 'local')
    }

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    
    print("\n=== Sentiment Analysis Service ===")
    print(f"Starting Flask application on port {port}...")
    print(f"Access the application at http://0.0.0.0:{port}")
    print("Press Ctrl+C to quit\n")
    
    # Run the app
    app.run(host='0.0.0.0', 
            port=port,
            debug=False)  # Set debug=False for production