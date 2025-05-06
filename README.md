# Automated Redirect Mapper

A Python tool that automatically generates URL redirect mappings for website migrations using URL structure analysis and Google's Gemini AI.

## 📋 Overview

This tool helps SEO professionals and web developers create redirect maps when migrating websites or restructuring URL hierarchies. It uses a multi-strategy approach:

1. **URL Structure Matching**: Analyzes URL patterns to find logical matches
2. **AI-Powered Matching**: Uses Google's Gemini AI for semantic understanding of URLs
3. **Smart Fallback**: Redirects to homepage when no good match is found

## ✨ Features

- **Multiple Matching Strategies**: Combines rule-based and AI approaches for better accuracy
- **Batch Processing**: Efficiently handles large URL sets with batched API calls
- **Caching System**: Reduces API costs and speeds up repeat processing
- **Multiple Export Formats**: Supports CSV, JSON, Nginx, and Apache formats
- **Analytics Report**: Provides insight into redirect quality and confidence
- **Google Colab Integration**: Runs easily in Google Colab for accessibility

## 🔧 Requirements

- Python 3.6+
- Google Gemini API key
- Required packages:
  - `google-generativeai`
  - `pandas`
  - `tqdm`
  - `requests`

## 📝 Step-by-Step Usage Instructions

### Option 1: Run in Google Colab (Recommended)

1. **Set Up Google Colab**:
   - Create a new Google Colab notebook
   - Copy and paste the script into a code cell or upload the `.py` file
   - Run the cell to initialize the script

2. **Set Up API Key**:
   - Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
   - Store the API key in Colab Secrets with the name `GEMINI_API_KEY` or be ready to enter it when prompted

3. **Prepare Your URL Files**:
   - Create two separate files:
     - **Old URLs file**: Contains the URLs of your current/old website
     - **New URLs file**: Contains the URLs of your new website structure
   - Files can be in CSV, TXT, or JSON format

4. **Run the Tool**:
   - Execute the script in Colab
   - Upload your old URLs file when prompted
   - Upload your new URLs file when prompted
   - Wait for processing to complete

5. **Download Results**:
   - The tool will automatically download:
     - `redirects.csv`: CSV format of redirect mappings
     - `redirects.json`: JSON format of redirect mappings
     - `redirects.nginx`: Nginx server configuration
     - `redirects.apache`: Apache server configuration
     - `redirect_analytics.json`: Analysis report
     - `redirect_mapper_cache.pkl`: Cache file for future runs

### Option 2: Run Locally

1. **Install Requirements**:
   ```bash
   pip install google-generativeai pandas tqdm requests
   ```

2. **Download the Script**:
   - Save the script as `redirect_mapper.py`

3. **Run from Command Line**:
   ```bash
   python redirect_mapper.py \
     --api-key "YOUR_GEMINI_API_KEY" \
     --old-urls old_urls.csv \
     --new-urls new_urls.csv \
     --output redirects.csv \
     --format csv \
     --report
   ```

4. **Command Line Options**:
   - `--api-key`: Your Google Gemini API key (required)
   - `--old-urls`: File with old URLs (required)
   - `--new-urls`: File with new URLs (required)
   - `--output`: Output file name (default: redirects.csv)
   - `--format`: Output format - csv/json/nginx/apache (default: csv)
   - `--report`: Generate analytics report
   - `--batch-size`: Number of URLs to process in each API call (default: 25)
   - `--no-cache`: Disable caching mechanism

## 📊 Understanding the Results

The tool generates redirect mappings with the following information:

1. **old_url**: The original URL that needs to be redirected
2. **new_url**: The destination URL where users should be redirected
3. **confidence**: A score (0-1) indicating how confident the tool is about the match
4. **reason**: The method used to create the mapping:
   - "URL structure match": Based on path similarities
   - "AI mapping": Based on Gemini AI's semantic understanding
   - "Homepage fallback": No good match found, redirect to homepage

### Confidence Score Interpretation:
- **High (0.9-1.0)**: Very reliable match
- **Medium (0.7-0.9)**: Good match
- **Low (<0.7)**: Less reliable match or fallback

## 🔄 How It Works

1. **URL Extraction**:
   - Reads URLs from input files
   - Normalizes URLs for consistent comparison

2. **URL Structure Analysis**:
   - Groups URLs by path depth and categories
   - Identifies patterns in URL structures

3. **URL Structure Matching**:
   - Compares path segments between old and new URLs
   - Calculates similarity scores based on path structure and keywords
   - Creates matches for URLs with high structural similarity

4. **AI-Powered Matching**:
   - For URLs without good structure matches, uses Google Gemini AI
   - Sends batches of URLs to reduce API costs
   - Gets AI-suggested matches based on semantic understanding

5. **Fallback Strategy**:
   - For URLs with no good matches, redirects to homepage
   - Applied to low-confidence AI matches

6. **Result Generation**:
   - Combines results from all strategies
   - Exports in requested format(s)
   - Generates analytics report

## 🚀 Advanced Usage

### Adjusting Thresholds

You can modify these values in the code to tune the matching behavior:

```python
self.url_structure_threshold = 0.7  # Threshold for URL structure matching
self.ai_confidence_threshold = 0.7  # Threshold for AI matching before fallback
```

- Higher thresholds = more fallbacks to homepage (safer but less specific)
- Lower thresholds = more specific matches (but potentially less accurate)

### Using the Cache

The tool creates a cache file (`redirect_mapper_cache.pkl`) to store API responses. This speeds up future runs and reduces API costs. To use a previously generated cache:

1. Place the cache file in the same directory as the script
2. Run the tool normally - it will automatically detect and use the cache

### Custom URL Structure Matching

For advanced users, you can modify the `url_structure_matching` method to implement custom matching logic based on your specific URL patterns and needs.

## 📈 Interpreting the Analytics Report

The analytics report (`redirect_analytics.json`) provides insights into the quality of your redirect mapping:

```json
{
  "total_redirects": 100,
  "reason_distribution": {
    "URL structure match": 60,
    "AI mapping": 30,
    "Homepage fallback (low confidence)": 10
  },
  "avg_confidence": 0.85,
  "high_confidence_count": 70,
  "medium_confidence_count": 20,
  "fallback_count": 10
}
```

Use this report to:
- Assess the overall quality of your redirect map
- Identify how many URLs need manual review (low confidence matches)
- Understand which matching methods were most effective

## ⚠️ Limitations

- The tool works best when old and new URLs have some structural similarities
- API rate limits may affect processing time for large URL sets
- Homepage fallbacks should be reviewed manually when possible
- The Gemini API may incur costs for large URL sets

## 🔒 Data Privacy

This tool processes URLs locally except when using the Gemini API. Only URL paths (not content) are sent to the API. No other data is transmitted outside your environment.

## 📄 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
