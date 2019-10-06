# Disaster Response Pipeline Project

## Introduction

This project is part of the Udacity DataScientist Nanodegree Program.

The aim of the project is to properly classify messages collected during
disasters (such as an earthquake, floods, ...) in order to send the appropriate
teams and/or goods.

## Project description

### Steps:

Overall 2 main steps have been performed for this project:

1- Load and clean data from 2 separate sources that contain the messages and
the classification, aka ELT Pipeline

2- Using this cleaned data, build and evaluate a model to classify new messages
aka Machine Learning Pipeline

### Files

According to the steps, 2 main python files were created:
- `process_data.py` which performs the ETL sub-Steps
- `train_classifier.py` which performs the machine learning part  

### Results and discussion

The training data is characterized with some categories that are very seldom.
This makes a model with high accuracy relatively simple to achieve (e.g. if a
  category only happens in 1% od the messages, a default model not assigning
  the category to the message is by default 99% accurate, which does not help
  us much).

In this case we want to have a high precision or positive predictive value.

I'm not quite happy with the performance of the model; however could not find
other methods to improve it significantly (using feature union with 2 new
  features and GridSearchCV only marginally improved the model).

In the end, I propose here a basic model (a kind of minimal viable product)
using AdaBoost as the classifier.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
