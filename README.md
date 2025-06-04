# Sentiment Analysis of YouTube Comments

This project performs sentiment classification on a dataset of YouTube comments using supervised, unsupervised, and rule-based machine learning techniques. It aims to detect whether a comment expresses a **Positive**, **Negative**, or **Neutral** sentiment by leveraging structured metadata and text-based features. The analysis also includes behavioral insights extracted via clustering and association rule mining.

## Objective

- Classify YouTube comments into sentiment categories  
- Handle real-world text data with informal language, slang, and emojis  
- Evaluate and compare multiple classification algorithms  
- Apply unsupervised clustering and association rule mining for deeper pattern discovery  

## Technologies & Libraries

- **Language**: Python  
- **Data Processing**: Pandas, NumPy  
- **Machine Learning**:  
  - Supervised: Random Forest, SVM, Logistic Regression  
  - Unsupervised: KMeans Clustering  
  - Class Balancing: SMOTE (from imbalanced-learn)  
  - Hyperparameter Tuning: GridSearchCV  
- **Association Rule Mining**: mlxtend (Apriori algorithm)  
- **Dimensionality Reduction**: PCA  
- **Visualization**: Matplotlib, Seaborn  

## Dataset Overview

- Comments with metadata: likes, replies, user type, timestamps, hashtags, etc.  
- Sentiment labels: Positive, Neutral, Negative  
- Simulated real-world YouTube data with imbalanced classes  

## Workflow

- Cleaned and preprocessed data by handling missing values and duplicates  
- Encoded categorical variables and scaled numerical features  
- Split the dataset into training and test sets  
- Balanced class distribution using SMOTE  
- Trained and evaluated multiple supervised models: Random Forest, SVM, and Logistic Regression  
- Tuned model hyperparameters using GridSearchCV  
- Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC  
- Applied KMeans clustering for unsupervised grouping and evaluated using Silhouette Score  
- Reduced dimensions with PCA for cluster visualization  
- Extracted association rules using the Apriori algorithm for interpretable sentiment patterns  

## Results & Insights

- Random Forest outperformed others with highest accuracy and interpretability  
- SMOTE significantly improved classification for minority sentiment classes  
- ROC-AUC curves showed strong class separability  
- KMeans clustering showed limited effectiveness (Silhouette Score = 0.14)  
- Apriori rules revealed behavioral patterns such as `{Brand} → {Link}` linked to positive sentiment

## Key Visualizations

- Confusion Matrices  
- Classification Reports  
- ROC Curves  
- Feature Importance (Random Forest)  
- Silhouette Score Visualization  
- PCA-based Cluster Plots  
- Association Rule Table (Lift, Confidence, Support)

## Limitations

- Sentiment detection is based on metadata only; no raw comment text was analyzed  
- Sarcasm, irony, and contextual nuance remain hard to detect  
- Clustering performance limited by lack of semantic depth in features

## Future Work

- Integrate textual sentiment features using TF-IDF, BERT, or LSTM  
- Combine metadata and NLP for hybrid modeling  
- Deploy the classifier as a web app using Flask or Streamlit  
- Add toxicity detection and moderation flags for comments

## License

This project is for academic and educational purposes only. You are welcome to reuse and extend it with proper attribution.
