This codebase is a Streamlit web application that serves as an interactive demonstration for a presentation on Natural Language Processing (NLP).

Here's a breakdown of what the application does:

*   **Lemmatization and Tokenization:** The app provides simple tools to demonstrate basic NLP tasks. You can input text to see how it's broken down into individual tokens (words) and how those words are reduced to their root form (lemmatization).
*   **Text Similarity Visualization:** The application can take two words or phrases and show how similar they are to each other, as well as to the words "king," "queen," "cat," and "dog." It does this by representing the words as mathematical vectors and then plotting them on a 2D graph. Words with similar meanings will appear closer together.
*   **Job Description Analysis:** The most significant part of the application is its ability to analyze a dataset of job descriptions.
    *   It visualizes the entire dataset of job titles on a 2D scatter plot, allowing you to see how different jobs cluster based on the language in their descriptions.
    *   It provides a search function where you can select a job title and the application will return the top 10 most similar job titles from the dataset.

**How it works:**

The application is built using several popular Python libraries for data science and NLP:

*   **Streamlit:** Creates the user interface for the web application.
*   **spaCy:** A powerful NLP library used for the tokenization, lemmatization, and for converting words into the vector representations needed for the similarity comparisons.
*   **scikit-learn:** A machine learning library used for dimensionality reduction (`t-SNE` and `PCA`) to create the 2D visualizations, and for calculating the `cosine similarity` to find similar job descriptions.
*   **Pandas and NumPy:** Used for handling and manipulating the data.
*   **Plotly:** Used to generate the interactive plots.

The data for the job descriptions is stored in the `reduced.pkl` file, which contains the job titles and their pre-calculated vector representations.

In essence, this codebase provides a hands-on way to understand and explore several key NLP concepts.
