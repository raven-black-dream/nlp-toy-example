# NLP Presentation Demo

This project is an interactive web application built with Streamlit to demonstrate key concepts in Natural Language Processing (NLP). It was created to serve as a visual aid for an NLP presentation.

## Features

The application showcases the following NLP tasks:

*   **Lemmatization**: Reduce words to their base or dictionary form.
*   **Tokenization**: Break down text into individual words or tokens.
*   **Text Similarity**: Visualize the semantic similarity between words and phrases using word embeddings and PCA for dimensionality reduction.
*   **Job Description Analysis**:
    *   Explore a dataset of job descriptions visualized in a 2D space using t-SNE.
    *   Find the most similar job titles to a selected one using cosine similarity.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd nlp_presentation-main
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  The application uses spaCy's `en_core_web_lg` model. The application will attempt to download it automatically on first run. If you encounter issues, you can download it manually:
    ```bash
    python -m spacy download en_core_web_lg
    ```

## Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run example_nlp_app.py
```

This will open the application in your web browser.

## Technologies Used

*   **Streamlit**: For creating the web application and user interface.
*   **spaCy**: For advanced NLP tasks like tokenization, lemmatization, and word embeddings.
*   **scikit-learn**: For machine learning algorithms like t-SNE, PCA, and cosine similarity.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Plotly**: For creating interactive visualizations.
