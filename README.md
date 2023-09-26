# Quantitative Stock Trading Strategy Development with Machine Learning

This repository contains the project work titled "Quantitative Stock Trading Strategy Development with Machine Learning" conducted as part of the STAT3799 Directed Studies in Statistics course at The University of Hong Kong.

BSc Thesis Project, HKU, HKSAR – (Jan 2022 – May 2022)

• Objectives: Developed a quantitative stock trading strategy using ML to predict stock prices and
analyze financial news sentiment.

• Methods: Used technical indicators like RSI and benchmark stock prices like S&P500 to predict
prices with ML models including MLP, RNN, LSTM, ARIMA, etc. Conducted sentiment analysis of financial news with a large pre-trained language model. Combined price and sentiment analysis into a trading strategy.

• Outcomes: Evaluated through backtesting, achieving higher ROI than the B&S benchmark.

## Introduction

The prediction of stock market returns has always been a controversial topic in the financial market. This project aims to make predictions on the stock market return (Hong Kong) using machine learning algorithms. The historical data of the stock in the Hong Kong stock market from 2016 till now is gathered from Yahoo Finance, covering approximately 6.5 years of data.

Two aspects of the possible influential factors on the prediction are considered. Firstly, the external impact of the target individual stock from the whole stock market and relevant industries and sectors is analyzed. This includes examining how the trend of the stock market and the stock indices for relevant industries and sectors influence the stock price of the individual stock. Secondly, the internal impact of the target individual stock is studied by using its historical data (opening price, closing price, trading volume) to make predictions.

The machine learning algorithms used in this project include:

- Linear Regression
- LASSO
- Random Forest
- Multi-Layer Perceptron (MLP)
- Recurrent Neural Network (RNN)
- Principal Component Analysis (PCA)

All algorithms used belong to the regression method, allowing for the prediction of the exact value of the stock price. The trading strategy can be developed based on these predictions.

## Repository Structure

The repository is organized as follows:

- `code`: This folder contains the code implementation of the machine learning algorithms and the data pre-processing steps.
- `docs`: This folder contains the written report of the project (PDF format).

## Results and Discussion

The results of the prediction models and the trading strategy implemented based on the forecasts are discussed in detail in the written report available in the `docs` folder.

## Acknowledgements

Special thanks to Dr. Zhang, Zhiqiang for supervising this project and providing valuable guidance throughout the study.

## References

The references are in the written report available in the `docs` folder.

Feel free to modify and customize this template to suit your specific needs. Good luck with your GitHub website!

