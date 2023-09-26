# Sentiment Score/Label from a list of news for a single stock

from transformers import pipeline
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
import dateutil.parser
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

## Use a pipeline as a high-level helper
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def get_weighted_sentiment(news_list, k):
    # Get sentiment predictions for each news article
    # Input: a list of news
    results = pipe(news_list)
    
    # Sort the results by confidence score in descending order, top k
    n = k
    if len(news_list) <= n:
        sorted_results = results
    else:
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:n]
        length_news_selected = n
    
    print('number of top news:', len(sorted_results))
    
    # Get a pie chart
    label_counts = {}
    for dictionary in sorted_results:
        label = dictionary['label']
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    
    plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%')
    plt.title('Label Distribution')

    # Calculate weighted average sentiment score: confidence score as weight, label as score
    total_score = 0.0
    #total_weight = 0.0
    for result in sorted_results:
        score = result['score']
        label = result['label']
        weight = score
        if label == 'positive':
            weight *= 1.0
        elif label == 'negative':
            weight *= -1.0
        total_score += weight
        #total_weight += score
    print("total score:", total_score)
    #weighted_average_score = total_score / total_weight
    average_score = total_score / len(sorted_results)
    print("average score:", average_score)
    
    #return weighted_average_score
    if average_score == 0.5:
        final_label = "neutral"
    elif average_score > 0.5:
        final_label = "positive"
    elif average_score < 0.5:
        final_label = "negative"
    return final_label

def get_weighted_sentiment_score(news_list, k):
    # Get sentiment predictions for each news article
    # Input: a list of news
    results = pipe(news_list)
    
    # Sort the results by confidence score in descending order, top k
    n = k

    if len(news_list) <= n:
        sorted_results = results
    else:
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:n]
        length_news_selected = n
    
    print('number of top news:', len(sorted_results))

    # Calculate weighted average sentiment score: confidence score as weight, label as score
    total_score = 0.0
    #total_weight = 0.0
    for result in sorted_results:
        score = result['score']
        label = result['label']
        weight = score
        if label == 'positive':
            weight *= 1.0
        elif label == 'negative':
            weight *= -1.0
        total_score += weight
        #total_weight += score
    print("total score:", total_score)
    #weighted_average_score = total_score / total_weight
    average_score = total_score / len(sorted_results)
    print("average score:", average_score)
    
    return average_score

def score_to_label(score):
    #score: integer
    if score > 0.5:
        label = "positive"
    elif score < 0.5:
        label = "negative"
    else:
        label = "neutral"
    return label

## Get news and pair with stock
api_key = "afa4c63381e545c185cbedd3917691b2"

def get_stock_news(ticker):
    # ticker: stock ticker, e.g., 'TSLA'; data: YY/MM/DD, e.g., '2023/07/01'

    # Create a connection to the Alpha Vantage API.
    api_key = "afa4c63381e545c185cbedd3917691b2"
    base_url = "https://www.alphavantage.co/query"

    # Get the stock quote for the given stock ticker.
    #params = {"function": "TIME_SERIES_DAILY", "symbol": ticker, "apikey": api_key}
    #response = requests.get(base_url, params=params)
    #data = json.loads(response.text)

    # Get the latest news articles for the given stock ticker.
    news_url = "https://newsapi.org/v2/everything"
    #params = {"q": ticker, "from": "2023-06-01", "apikey": api_key}
    params = {"q": ticker, "apikey": api_key}
    response = requests.get(news_url, params=params)
    data = json.loads(response.text)

    # Create a dataframe with the data and time of the news articles.
    articles = data["articles"]
    df = pd.DataFrame(articles)
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["publishedAt"])
    df["time"] = df["date"].dt.time
    df["date"] = df["date"].dt.strftime("%Y/%m/%d")

    # Return the dataframe.
    return df

def exact_date(ticker, date, r):
    # k: random k news
    df = get_stock_news(ticker)

    # Select the dataframe with the specific date.
    df = df[df["date"] == date]

    # If the number of rows in the dataframe is greater than 10, we random choose ten rows.
    num = r
    if len(df) > num:
        df = df.sample(n=num)
        print('number of random news:', num)
    else:
        print('number of random news:', len(df))

    return df

#ticker = 'TSLA'
#date = '2023/07/13'
##random_news
#r = 10
##top_news based on confidence score
#t = 5
def label(ticker, date, r, t):
    example = exact_date(ticker, date, r)
    # Use content, title and description of a article for the sentiment analysis
    content = example["content"].tolist()
    title = example["title"].tolist()
    description = example["description"].tolist()
    text_list = content+title+description
    text_list = list(filter(lambda x: x is not None, text_list))
    if len(text_list) == 0:
        pass
    elif len(text_list) <= t:
        return get_weighted_sentiment(text_list, len(text_list))
    else:
        return get_weighted_sentiment(text_list, t)

def score(ticker, date, r, t):
    example = exact_date(ticker, date, r)
    # Use content, title and description of a article for the sentiment analysis
    content = example["content"].tolist()
    title = example["title"].tolist()
    description = example["description"].tolist()
    text_list = content+title+description
    text_list = list(filter(lambda x: x is not None, text_list))
    if len(text_list) == 0:
        pass
    elif len(text_list) <= t:
        return get_weighted_sentiment_score(text_list, len(text_list))
    else:
        return get_weighted_sentiment_score(text_list, t)

# A ticker is simply the stock code.
# Dates are in YY/MM/DD format.
# random_news (r) specifies the number of news articles you want to randomly scrape from the website for a given stock on a given date.
# top_news (t) specifies the number of news articles you want to use for sentiment analysis, selected from all of the randomly scraped news articles for a given stock on a given date.

def str_to_date(str_date):
    #input string
    #convert string to date
    date_object = dateutil.parser.parse(str_date)
    #leave time keep date
    date = date_object.date()

    return date

def date_to_str(date):
    #input date
    #convert date to str
    output = date.strftime("%Y/%m/%d")

    return output

def previous_dates(date, num):
    #date: the current date, string
    #num: no of previous dates we want, integer

    delta = datetime.timedelta(days=1)

    current_date = str_to_date(date)
    dates_str = []
    n = 0

    while n < num:
        previous_date = current_date - delta
        current_date = previous_date
        dates_str.append(date_to_str(previous_date))
        n += 1
    print("dates considered:", dates_str)
    return dates_str

def exact_n_previous_dates(ticker, date, r, n):
    # r: random r news
    # n: no of previous dates
    df = get_stock_news(ticker)

    dates = previous_dates(date, n)

    # Select the dataframe with the specific date.
    df = df[df["date"].isin(dates)]

    # If the number of rows in the dataframe is greater than 10, we random choose ten rows.
    num = r
    if len(df) > num:
        df = df.sample(n=num)
        print('number of random news:', num)
    else:
        print('number of random news:', len(df))

    # Return the news in n previous dates
    return df

#ticker = 'TSLA'
#date = '2023/07/13'
##random_news
#r = 10
##top_news based on confidence score
#t = 5
def label_with_previous_n_dates(ticker, date, r, t, n):
    # n: no of previous dates
    example = exact_n_previous_dates(ticker, date, r, n)
    # Use content, title and description of a article for the sentiment analysis
    content = example["content"].tolist()
    title = example["title"].tolist()
    description = example["description"].tolist()
    text_list = content+title+description
    text_list = list(filter(lambda x: x is not None, text_list))
    if len(text_list) == 0:
        pass
    elif len(text_list) <= t:
        return get_weighted_sentiment(text_list, len(text_list))
    else:
        return get_weighted_sentiment(text_list, t)

def score_with_previous_n_dates(ticker, date, r, t, n):
    example = exact_n_previous_dates(ticker, date, r, n)
    # Use content, title and description of a article for the sentiment analysis
    content = example["content"].tolist()
    title = example["title"].tolist()
    description = example["description"].tolist()
    text_list = content+title+description
    text_list = list(filter(lambda x: x is not None, text_list))
    if len(text_list) == 0:
        pass
    elif len(text_list) <= t:
        return get_weighted_sentiment_score(text_list, len(text_list))
    else:
        return get_weighted_sentiment_score(text_list, t)

# The latest available date for news is two days before the current day.
# For example, on July 14, 2023, the latest available news date would be July 12, 2023.

def get_sentiment_label(beginning_date, end_date, ticker, r, t, n):
  """Creates a trend graph that shows the sentiment label for each date in the 1-month period.

  Args:
    beginning_date: The beginning date of the 1-month period.
    end_date: The end date of the 1-month period.
    ticker = 'TSLA'
    date = '2023/07/13'
    #random_news
    r = 10
    #top_news based on confidence score
    t = 5

  Returns:
    A plot object that shows the trend graph.
  """

  delta = datetime.timedelta(days=1)

  current_date = str_to_date(beginning_date)
  dates_str = []
  dates_date = []
  while current_date <= str_to_date(end_date):
    dates_str.append(date_to_str(current_date))
    dates_date.append(current_date)
    current_date += delta
  
  sentiment_labels = []
  for date in dates_str:
    print(date)
    sentiment_labels.append(label_with_previous_n_dates(ticker, date, r, t, n))
  
  # Create a sample DataFrame
  sentiment = {
    'Date': dates_date,
    'sentiment': sentiment_labels
  }
    
  df = pd.DataFrame(sentiment)

  return df

## Visualization
def get_sentiment_score(beginning_date, end_date, ticker, r, t, n):
  """Creates a trend graph that shows the sentiment label for each date in the 1-month period.

  Args:
    beginning_date: The beginning date of the 1-month period.
    end_date: The end date of the 1-month period.
    ticker = 'TSLA'
    date = '2023/07/13'
    #random_news
    r = 10
    #top_news based on confidence score
    t = 5

  Returns:
    A plot object that shows the trend graph.
  """

  delta = datetime.timedelta(days=1)

  current_date = str_to_date(beginning_date)
  dates_str = []
  dates_date = []
  while current_date <= str_to_date(end_date):
    dates_str.append(date_to_str(current_date))
    dates_date.append(current_date)
    current_date += delta
  
  sentiment_labels = []
  for date in dates_str:
    print(date)
    sentiment_labels.append(score_with_previous_n_dates(ticker, date, r, t, n))
  
  # Create a sample DataFrame
  sentiment = {
    'Date': dates_date,
    'sentiment': sentiment_labels
  }
    
  df = pd.DataFrame(sentiment)

  return df

def lineplot_sentiment(df):
    
  # Create a line plot
  plt.plot(df['Date'], df['sentiment'], label='Sentiment')

  # Add a title, axes labels and legend to the plot
  plt.title('Sentiment Scores')
  plt.xlabel('Date')
  plt.ylabel('Sentiment')
  plt.legend()

  # Set the x-axis limits
  plt.xticks(df['Date'])
  # Adjust the x-axis tick labels to fit within the plot size
  plt.gcf().autofmt_xdate()

  # Show the plot
  plt.show()

def barplot_sentiment(df):
    
  # Create a bar plot
  plt.bar(df['Date'], df['sentiment'], label='Sentiment')

  # Add a title, axes labels and legend to the plot
  plt.title('Sentiment Scores')
  plt.xlabel('Date')
  plt.ylabel('Sentiment')
  plt.legend()

  # Set the x-axis limits
  plt.xticks(df['Date'])
  # Adjust the x-axis tick labels to fit within the plot size
  plt.gcf().autofmt_xdate()
  # Adjust the y-limits of the bar plot
  # set ylim to be just above the highest value in y
  plt.ylim(min(df['sentiment']) - 1, max(df['sentiment']) + 1)

  # Show the plot
  plt.show()

### Link the sentiment score with price
def get_stock_price(start_date, end_date, ticker):
    # Set the start and end dates
    start_date = str_to_date(start_date)
    end_date = str_to_date(end_date)

    # Get the stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Extract the daily closing prices
    closing_prices = data['Close']

    df = closing_prices.to_frame('price')
    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    return df

def plot_stock(df):

    # Create a line plot
    plt.plot(df['Date'], df['price'], label='Stock Price')

    # Set axes labels
    plt.xlabel('Date')

    # Set the x-axis limits
    plt.xticks(df['Date'])

    # Set the title ,legend and axes labels
    plt.title('Stock Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()

    # Adjust the x-axis tick labels to fit within the plot size
    plt.gcf().autofmt_xdate()

    # Show the plot
    plt.show()

def merge_stock_with_sentiment_data(stock, sentiment):

    # Merge the two dataframes using a left join on the 'Date' column
    merged = pd.merge(sentiment, stock, on='Date', how='left')

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    merged[['sentiment', 'price']] = scaler.fit_transform(merged[['sentiment', 'price']])

    return merged

def plot_stock_with_sentiment(merged):
    
    # Create a line plot with two lines
    plt.plot(merged['Date'], merged['sentiment'], label='Sentiment')
    plt.plot(merged['Date'], merged['price'], label='Stock Price')

    # Add a title and legend to the plot
    plt.title('Sentiment Scores and Stock Prices')
    plt.legend()

    # Set axes labels
    plt.xlabel('Date')

    # Set the x-axis limits
    plt.xticks(merged['Date'])
    # Adjust the x-axis tick labels to fit within the plot size
    plt.gcf().autofmt_xdate()
    plt.ylim(-1, 2)

    # Show the plot
    plt.show()
