## Disaster Response Message Classification Pipelines


# Libraries
1. Project Description
2. File Description
3. Analysis
4. Results
5. Licensing, Authors, and Acknowledgements
6. Instructions

# Libraries
1. pandas
2. numpy
3. sqlalchemy
4. matplotlib
5. plotly
6. NLTK
7. NLTK [punkt, wordnet, stopwords]
8. sklearn
9. joblib
10. flask

# Project Description
Figure Eight Data Set: Disaster Response Messages provides thousands of messages that have been sorted into 36 categories. These messages are classified into specific categories such as Water, Hospitals, Related Help, which are specifically intended to assist emergency personnel in their relief efforts.

The main objective of this project is to create an application that can help emergency workers analyze incoming messages and classify them into specific categories to speed up aid and contribute to a more efficient distribution of people and other resources.

# File Description
There are three main folders:

1. Data
disaster_categories.csv: dataset including all the categories
disaster_messages.csv: dataset including all the messages
process_data.py: ETL pipeline scripts to read, clean, and save data into a database
DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data

2. Models
train_classifier.py: machine learning pipeline scripts to train and export a classifier
classifier.pkl: output of the machine learning pipeline, i.e. a trained classifier

3. App
run.py: Flask file to run the web application
templates contains html file for the web application

# Analysis
1. Data Preparation

Modify the Category csv; split each category into a separate column
Merge Data from the two csv files (messages.csv & categories.csv)
remove duplicates and any non-categorized valued
create SQL database DisasterResponse.db for the merged data sets

2. Text Preprocessing

Tokenize text
remove special characters
lemmatize text
remove stop words

3. Build Machine Learning Pipeline

Build Pipeline with countevectorizer and tfidtransformer
Seal pipeline with multioutput classifier with random forest
Train Pipeline (with Train/Test Split)
Print classification reports and accuracy scores

4. Improve Model

Preform GirdSearchCV
Find best parameters

5. Export Model as .pkl File

You could also use Joblib as it can be faster. read more here

# Results
Created an ETL pipeline to read data from two csv files, clean data, and save data into a SQLite database.
Created a machine learning pipeline to train a multi-output classifier on the various categories in the dataset.
Created a Flask app to show data visualization and classify any message that users would enter on the web page.


# Licensing, Authors, and Acknowledgements
Thanks to Udacity and FigureEight.

# Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves model python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

https://view6914b2f4-3001.udacity-student-workspaces.com/


