📧 Spam–Ham Detection System (Machine Learning + Streamlit)

This project implements a Spam vs Ham (Not Spam) detection system using Machine Learning (TF-IDF + Logistic Regression) and provides an interactive web interface built with Streamlit.
It supports both single message prediction and batch CSV prediction.

🚀 Features

✔️ Automatic detection of spam and ham messages

✔️ Handles different dataset formats (auto-detects text & label columns)

✔️ Uses TF-IDF vectorization with n-grams for better text representation

✔️ Balanced Logistic Regression to handle class imbalance

✔️ Streamlit web app with:

Single message prediction

Batch CSV upload & download

Prediction confidence scores

✔️ Model saved using Joblib for reuse

🧠 Machine Learning Approach
Model Pipeline
Text → TF-IDF Vectorizer → Logistic Regression → Prediction

Algorithms Used

TF-IDF Vectorizer

Unigrams + Bigrams

Converts text into numerical features

Logistic Regression

Class-weight balancing

Suitable for binary text classification

📁 Project Structure
spam-ham-detection/
│
├── data.csv                 # Training dataset
├── train.py                 # Model training script
├── app.py                   # Streamlit web application
├── spam_model.pkl           # Trained ML model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

📊 Dataset Requirements

The dataset should be a CSV file containing:

One column with text messages

One column with labels (spam, ham, 1, 0, etc.)

The training script automatically detects:

Text column (message content)

Label column (spam/ham)

Example formats supported:

label,text
spam,Win a free prize now!
ham,Are we meeting tomorrow?


or

v1,v2
spam,Call now to claim reward
ham,See you at 5 PM

🏋️ Model Training
Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Train the model
python train.py


This will:

Load data.csv

Clean and normalize text

Train the model

Evaluate accuracy & performance

Save the trained model as spam_model.pkl

🧪 Model Evaluation

During training, the script outputs:

Accuracy

Precision

Recall

F1-score

Confusion metrics (via classification report)

🌐 Streamlit Web Application
Run the app
streamlit run app.py

App Features
🔹 Single Message Mode

Enter any message

Get Spam/Ham prediction

View confidence score

🔹 Batch Mode

Upload CSV file

Auto-detect message column

Predict spam/ham for all rows

Download results as CSV

📦 Output Example
Message	Prediction	Confidence
Win money now!	Spam	98.2%
See you later	Ham	96.5%
🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Joblib

🔮 Future Improvements

Add deep learning models (LSTM / BERT)

Email header analysis

Multilingual spam detection

Deployment using Docker / Cloud

REST API integration

👨‍💻 Author

Shagun
Spam–Ham Detection Project
Machine Learning | NLP | Streamlit
