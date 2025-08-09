from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import json
from io import StringIO
import logging
from werkzeug.utils import secure_filename
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Try to import additional sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER sentiment analyzer not available. Install with: pip install vaderSentiment")

class TextCleaner:
    """Text cleaning and preprocessing utilities"""

    @staticmethod
    def remove_special_chars(text):
        """Remove special characters and symbols"""
        if not isinstance(text, str):
            return str(text)
        # Keep alphanumeric, spaces, and basic punctuation
        return re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)

    @staticmethod
    def normalize_text(text):
        """Normalize text: lowercase and trim whitespace"""
        if not isinstance(text, str):
            return str(text)
        return text.lower().strip()

    @staticmethod
    def is_valid_text(text, min_length=3):
        """Check if text is valid for analysis"""
        if not isinstance(text, str):
            return False
        text = text.strip()
        return len(text) >= min_length and not text.isspace()

    @staticmethod
    def clean_text(text, options):
        """Apply cleaning options to text"""
        if not isinstance(text, str):
            text = str(text)

        cleaned_text = text

        if options.get('removeSpecialChars', False):
            cleaned_text = TextCleaner.remove_special_chars(cleaned_text)

        if options.get('normalizeText', False):
            cleaned_text = TextCleaner.normalize_text(cleaned_text)

        return cleaned_text

class SentimentAnalyzer:
    """Advanced sentiment analysis with multiple methods"""

    @staticmethod
    def analyze_textblob(text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                sentiment = 'Positive'
            elif polarity < -0.1:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            return {
                'sentiment': sentiment,
                'score': round(polarity, 3),
                'confidence': round(abs(polarity), 3),
                'subjectivity': round(subjectivity, 3)
            }
        except Exception as e:
            logger.error(f"TextBlob analysis error: {e}")
            return {'sentiment': 'Neutral', 'score': 0, 'confidence': 0}

    @staticmethod
    def analyze_vader(text):
        """Analyze sentiment using VADER"""
        if not VADER_AVAILABLE:
            return SentimentAnalyzer.analyze_textblob(text)

        try:
            scores = vader_analyzer.polarity_scores(text)
            compound = scores['compound']

            if compound >= 0.05:
                sentiment = 'Positive'
            elif compound <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            return {
                'sentiment': sentiment,
                'score': round(compound, 3),
                'confidence': round(abs(compound), 3),
                'positive': round(scores['pos'], 3),
                'negative': round(scores['neg'], 3),
                'neutral': round(scores['neu'], 3)
            }
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return SentimentAnalyzer.analyze_textblob(text)

    @staticmethod
    def analyze_combined(text):
        """Combine TextBlob and VADER analysis"""
        textblob_result = SentimentAnalyzer.analyze_textblob(text)
        vader_result = SentimentAnalyzer.analyze_vader(text)

        # Average the scores
        avg_score = (textblob_result['score'] + vader_result['score']) / 2
        avg_confidence = (textblob_result['confidence'] + vader_result['confidence']) / 2

        # Determine final sentiment
        if avg_score > 0.1:
            sentiment = 'Positive'
        elif avg_score < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'sentiment': sentiment,
            'score': round(avg_score, 3),
            'confidence': round(avg_confidence, 3),
            'textblob_sentiment': textblob_result['sentiment'],
            'vader_sentiment': vader_result['sentiment']
        }

    @staticmethod
    def analyze(text, method='textblob', threshold=0.1):
        """Analyze sentiment using specified method"""
        if method == 'vader':
            return SentimentAnalyzer.analyze_vader(text)
        elif method == 'combined':
            return SentimentAnalyzer.analyze_combined(text)
        else:
            return SentimentAnalyzer.analyze_textblob(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single text sentiment prediction"""
    try:
        data = request.json
        user_input = data.get('text', '').strip()
        method = data.get('method', 'textblob')
        clean_options = data.get('clean', {})

        if not user_input:
            return jsonify({'error': 'No text provided'})

        # Clean text if requested
        if clean_options:
            user_input = TextCleaner.clean_text(user_input, clean_options)

        # Validate text
        if not TextCleaner.is_valid_text(user_input):
            return jsonify({'error': 'Text is too short or invalid for analysis'})

        # Analyze sentiment
        result = SentimentAnalyzer.analyze(user_input, method)

        logger.info(f"Single text analysis completed: {result['sentiment']}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during analysis'})

@app.route('/process_csv', methods=['POST'])
def process_csv():
    """Process CSV file for batch sentiment analysis"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Get parameters
        text_column = request.form.get('textColumn', 'text')
        method = request.form.get('method', 'textblob')
        cleaning_options = json.loads(request.form.get('cleaningOptions', '{}'))
        threshold = float(request.form.get('threshold', 0.1))

        # Read CSV file
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None

            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                return jsonify({'error': 'Unable to read CSV file. Please check the file encoding.'})

        except Exception as e:
            logger.error(f"CSV reading error: {e}")
            return jsonify({'error': 'Invalid CSV file format'})

        # Check if text column exists
        if text_column not in df.columns:
            available_columns = list(df.columns)
            return jsonify({
                'error': f'Column "{text_column}" not found. Available columns: {available_columns}'
            })

        # Get original data count
        original_count = len(df)
        logger.info(f"Processing {original_count} rows from CSV")

        # Clean and preprocess data
        df = preprocess_dataframe(df, text_column, cleaning_options)

        if df.empty:
            return jsonify({'error': 'No valid data remaining after preprocessing'})

        processed_count = len(df)
        logger.info(f"After preprocessing: {processed_count} rows remaining")

        # Perform sentiment analysis
        results = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for index, row in df.iterrows():
            text = str(row[text_column])

            try:
                analysis_result = SentimentAnalyzer.analyze(text, method, threshold)

                result = {
                    'text': text,
                    'sentiment': analysis_result['sentiment'],
                    'score': analysis_result.get('score'),
                    'confidence': analysis_result.get('confidence')
                }

                results.append(result)

                # Count sentiments
                if analysis_result['sentiment'] == 'Positive':
                    positive_count += 1
                elif analysis_result['sentiment'] == 'Negative':
                    negative_count += 1
                else:
                    neutral_count += 1

            except Exception as e:
                logger.error(f"Error analyzing row {index}: {e}")
                # Add as neutral if analysis fails
                results.append({
                    'text': text,
                    'sentiment': 'Neutral',
                    'score': 0,
                    'confidence': 0
                })
                neutral_count += 1

        # Prepare response
        response_data = {
            'results': results,
            'total_processed': len(results),
            'original_count': original_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'processing_info': {
                'method': method,
                'text_column': text_column,
                'cleaning_applied': cleaning_options,
                'threshold': threshold
            }
        }

        logger.info(f"CSV processing completed: {len(results)} results")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        return jsonify({'error': 'An error occurred while processing the CSV file'})

def preprocess_dataframe(df, text_column, cleaning_options):
    """Preprocess dataframe based on cleaning options"""
    try:
        # Make a copy to avoid modifying original
        df_clean = df.copy()

        # Remove empty rows if requested
        if cleaning_options.get('removeEmptyRows', True):
            df_clean = df_clean.dropna(subset=[text_column])
            df_clean = df_clean[df_clean[text_column].astype(str).str.strip() != '']
            df_clean = df_clean[df_clean[text_column].astype(str).str.len() >= 3]

        # Apply text cleaning
        if any(cleaning_options.values()):
            df_clean[text_column] = df_clean[text_column].apply(
                lambda x: TextCleaner.clean_text(str(x), cleaning_options)
            )

        # Remove duplicates if requested
        if cleaning_options.get('removeDuplicates', True):
            original_len = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=[text_column])
            logger.info(f"Removed {original_len - len(df_clean)} duplicate entries")

        # Final validation - remove any invalid texts
        df_clean = df_clean[df_clean[text_column].apply(TextCleaner.is_valid_text)]

        return df_clean

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return df

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
