# Sentiment Analysis Service

A Flask-based web service that provides sentiment analysis for text and bulk file processing. The service uses state-of-the-art natural language processing models to analyze sentiments in text.

## Features

- Single text sentiment analysis
- Bulk file processing (CSV, Excel, TXT)
- Interactive commands (hi, help, start)
- Visualization of sentiment distribution
- Support for various file formats
- Easy-to-use API endpoints

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd bulk_sentiment_clean
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Twilio credentials:
```
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Access the service at `http://localhost:5000`

3. Available commands:
- `hi` or `hello`: Get welcome message
- `help`: View usage instructions
- `start`: Begin analysis
- Send any text: Get sentiment analysis
- Upload file: Get bulk analysis

## API Endpoints

- `GET /`: Welcome endpoint
- `POST /analyze`: Sentiment analysis endpoint
  - Accept text input
  - Accept file upload
  - Returns sentiment analysis results

## File Support

Supported file formats:
- CSV
- Excel (xlsx, xls)
- Text files (txt)

## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies

## License

[Your chosen license] 