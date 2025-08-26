# SMS Spam Classification 📱🚫

A machine learning project that automatically classifies SMS messages as spam or legitimate (ham), helping users filter unwanted texts using Python and popular data science libraries.

## 🎯 About

SMS-Spam-Classification is a comprehensive machine learning project built using Python and Jupyter Notebook. The project demonstrates the complete workflow from data preprocessing to model training and evaluation, leveraging accessible data science tools to create an educational and practical spam detection solution.

## ✨ Features

- **Automated SMS spam filter** using popular machine learning algorithms
- **Step-by-step text preprocessing** including:
  - Text cleaning and normalization
  - Tokenization
  - Stop-word removal
  - Stemming/Lemmatization
- **Model building and evaluation** for accurate message classification
- **Sample dataset** with exploratory data analysis
- **Transparent workflow** designed for learning and easy extension
- **Performance metrics** to evaluate model effectiveness

## 🛠️ Technologies Used

- **Python 3.x** - Core programming language
- **Jupyter Notebook** - Interactive development environment
- **scikit-learn** - Machine learning algorithms and tools
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **nltk** - Natural language processing
- **matplotlib** - Data visualization
- **wordcloud** - Text visualization

## 📋 Prerequisites

Before running this project, make sure you have:

- Python 3.x installed on your system
- Jupyter Notebook (or Anaconda distribution)
- Git (for cloning the repository)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tramakrishna3012/SMS-Spam-Classification.git
cd SMS-Spam-Classification
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

**Alternative:** If you're using Anaconda, most dependencies are already included:

```bash
conda install scikit-learn pandas numpy nltk matplotlib
```

### 3. Download NLTK Data (if required)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Launch the Notebook

```bash
jupyter notebook SMS-Spam-Classification.ipynb
```

Or simply open the notebook in your preferred environment (VS Code, JupyterLab, etc.).

## 📖 Usage Guide

1. **Load the Dataset**: The notebook includes a sample dataset for training and testing
2. **Data Exploration**: Analyze the distribution of spam vs. ham messages
3. **Text Preprocessing**: Follow the step-by-step cells for:
   - Data cleaning
   - Text vectorization (TF-IDF, Bag of Words)
   - Feature engineering
4. **Model Training**: Train various machine learning models
5. **Model Evaluation**: Assess accuracy, precision, recall, and F1-score
6. **Prediction**: Use the trained model to classify new SMS messages

### Example Usage

```python
# After training the model
new_message = "Congratulations! You've won $1000. Click here to claim your prize!"
prediction = model.predict([new_message])
print("Spam" if prediction[0] == 1 else "Ham")
```

## 📊 Dataset

The project uses a publicly available SMS spam dataset containing:
- **Ham messages**: Legitimate SMS messages
- **Spam messages**: Unwanted promotional or fraudulent texts
- **Features**: Message text and corresponding labels

## 🤖 Machine Learning Pipeline

1. **Data Loading & Exploration**
2. **Text Preprocessing**
   - Lowercasing
   - Removing special characters
   - Tokenization
   - Stop words removal
   - Stemming/Lemmatization
3. **Feature Extraction**
   - TF-IDF Vectorization
   - Bag of Words
4. **Model Training**
   - Naive Bayes
   - Support Vector Machine
   - Random Forest
   - Logistic Regression
5. **Model Evaluation**
   - Confusion Matrix
   - Classification Report
   - ROC Curve Analysis

## 📈 Model Performance

The trained models achieve high accuracy in distinguishing between spam and legitimate messages. Detailed performance metrics are provided in the notebook including:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Reliability of spam predictions
- **Recall**: Ability to identify all spam messages
- **F1-Score**: Balanced measure of precision and recall

## 🔧 Customization

You can extend this project by:

- **Adding new features** like message length, number of capital letters, etc.
- **Experimenting with different algorithms** like deep learning models
- **Using different vectorization techniques** like Word2Vec or BERT embeddings
- **Implementing real-time prediction** with a web interface
- **Adding multilingual support** for non-English messages

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Trama Krishna**
- GitHub: [@tramakrishna3012](https://github.com/tramakrishna3012)

## 🙏 Acknowledgments

- Thanks to the open-source community for providing excellent libraries
- Dataset providers for making SMS spam data publicly available
- Contributors and users who help improve this project

## 📞 Support

If you have any questions or run into issues, please:
- Open an issue on GitHub
- Check the existing issues for solutions
- Feel free to reach out for collaboration opportunities

---

⭐ **If you found this project helpful, please give it a star!** ⭐
