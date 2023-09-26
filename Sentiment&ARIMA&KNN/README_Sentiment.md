# Sentiment Analysis for Stock Trading Strategy

This code is for sentiment analysis of news articles for a single stock. It uses the pre-trained language model `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` in `transformers` library to get sentiment predictions for each news article, and then calculates a weighted average sentiment score based on the confidence score of each prediction. The latest available date for news is two days before the current day. For example, on July 14, 2023, the latest available news date would be July 12, 2023. The oldest available date for news is 1 month before the current day. For example, on July 14, 2023, the oldest availabel news date would be Jun 14, 2023. Then the sentiment score for making the trading strategy at the target day is based on past 5 days' news before the target day.

## Flow

How to generate the sentiment label for making the trading strategy at the target day: (flow)
    Select a target stock (ticker)
    Select the target date (date)
    Based on past N days' news before the target day (N)
    Randomly scrape R news articles from the website for a given stock on a given date with Content + Title + Description as text list (R)
    Select top T texts from text list based on confidence score (T)
    Do sentiment analysis on selected texts
    Get sentiment label with confidence score for each text
    Calculate the weighted average sentiment score (WASE)
    Convert label to score: `positive/negative/neutral = 1/-1/0`
    Confidence score as weight
    `WASE = sum(score*confidence score)/number of texts`
    Conservatively convert score to label: `>0.5/<0.5/=0.5 = ‘positive’/’negative’/’neutral’`

Trading strategy: based on sentiment label
    `‘positive’: buy signal`
    `‘negative’: sell signal`
    `‘neutral’:  no action`


## Usage
To use this code, first install libraries and import functions in your enviroment:

```
from transformers import pipeline
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
import dateutil.parser
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
```

Then, run the following code in your terminal:

```
from sentiment import score_with_previous_n_dates
from sentiment import score_to_label

ticker = 'TSLA' #string
date = '2023/07/13' #string
r = 50 #int
t = 20 #int
n = 5 #int

label = score_to_label(score_with_previous_n_dates(ticker, date, r, t, n))
label
```

A ticker is simply the stock code. Dates are in YY/MM/DD format. random_news (r) specifies the number of news articles you want to randomly scrape from the website for a given stock on a given date. top_news (t) specifies the number of news articles you want to use for sentiment analysis, selected from all of the randomly scraped news articles for a given stock on a given date. previous_dates (n) specifies the number of previous days before the current day you want scratch news to use for sentiment analysis.


This will print the weighted average sentiment label for the given stock and date.

Or you directly use the `sentiment.ipynb`.

## Others

There are many other functions, for sentiment score analysis. You can also get a lot of visulizations provided with many defined functions. You can view the `sentiment.ipynb` for details.