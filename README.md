# Semantic Product Review Analyzer

A powerful NLP-based tool for analyzing product reviews using semantic understanding and sentiment analysis.

## 🚀 Features

- **Aspect Extraction**: Automatically identifies product aspects (e.g., camera, battery, price) from reviews
- **Sentiment Analysis**: Determines sentiment (positive/negative/neutral) for each aspect
- **Interactive Dashboard**: Beautiful Streamlit-based UI with visualizations
- **Word Cloud**: Visual representation of most frequent terms
- **Real-time Analysis**: Get instant insights from your product reviews

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semantic-review-analyzer.git
   cd semantic-review-analyzer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required NLTK data and spaCy model:
   ```bash
   python -m nltk.downloader punkt stopwords wordnet
   python -m spacy download en_core_web_sm
   ```

## 🚦 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter a product review in the text area and click "Analyze Review"

## 🧠 How It Works

1. **Text Preprocessing**: The input text is cleaned and normalized
2. **Aspect Extraction**: Key product aspects are identified using NLP techniques
3. **Sentiment Analysis**: Each aspect is analyzed for sentiment using a fine-tuned BERT model
4. **Visualization**: Results are presented with interactive charts and visualizations

## 📊 Example Input/Output

**Input:**
"The camera quality is amazing with great low-light performance, but the battery drains too quickly."

**Output:**
- **Overall Sentiment:** Positive
- **Aspects:**
  - Camera: 😊 Positive (0.95)
  - Battery: 😞 Negative (0.85)

## 📁 Project Structure

```
semantic-review-analyzer/
├── app.py               # Streamlit application
├── semantic_analyzer.py # Core NLP processing logic
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
