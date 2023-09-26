setwd("~/Desktop/HKU/Academic/Y4S2/Capstone-STAT3799")

library(tseries); library(magrittr); library(dplyr)
library(xts); library(TTR); library(randomForest); library(readxl); library(writexl)



## Prediction


RFpred <- function(stock_select, rffit) {
    # Require library for technical indicators
    library(TTR) 
    price <- data2$Close
    
    # RSI: Relative strength Index
    rsi30 <- function(price) {
        rsi1 <- RSI(price,n=2)
        rsi2 <- RSI(price,n=3)
        rsi3 <- RSI(price,n=5)
        rsi4 <- RSI(price,n=10)
        rsi5 <- RSI(price,n=15)
        rsi6 <- RSI(price,n=20)
        rsi7 <- RSI(price,n=30)
        rsi <- cbind(rsi1,rsi2,rsi3,rsi4,rsi5,rsi6,rsi7)
        names(rsi) <- paste(c("RSI2","RSI3","RSI5","RSI10",
                              "RSI15","RSI20","RSI30"))
        return(rsi)
    }
    rsi <- rsi30(price)
    
    # MACD: Moving Average Convergence/Divergence
    macd30 <- function(price) {
        macd1 <- MACD(price,nFast = 3,nSlow = 5)[,1]
        macd2 <- MACD(price,nFast = 3,nSlow = 10)[,1]
        macd3 <- MACD(price,nFast = 3,nSlow = 20)[,1]
        macd4 <- MACD(price,nFast = 5,nSlow = 10)[,1]
        macd5 <- MACD(price,nFast = 5,nSlow = 20)[,1]
        macd6 <- MACD(price,nFast = 5,nSlow = 30)[,1]
        macd7 <- MACD(price,nFast = 10,nSlow = 20)[,1]
        macd8 <- MACD(price,nFast = 10,nSlow = 30)[,1]
        macd9 <- MACD(price,nFast = 12,nSlow = 26)[,1]
        macd <- cbind(macd1,macd2,macd3,macd4,macd5,
                      macd6,macd7,macd8,macd9)
        names(macd) <- paste(c("MACD35","MACD310","MACD320",
                               "MACD510","MACD5210","MACD530",
                               "MACD1020","MACD1030","MACD1226"))
        return(macd)
    }
    macd <- macd30(price)
    
    # SMA: Simple Moving Average, normalized by 30-day maximum
    sma30 <- function(price) {
        max30 <- runMax(price,30)
        sma1 <- runMean(price,2)/max30
        sma2 <- runMean(price,3)/max30
        sma3 <- runMean(price,5)/max30
        sma4 <- runMean(price,10)/max30
        sma5 <- runMean(price,15)/max30
        sma6 <- runMean(price,20)/max30
        sma7 <- runMean(price,30)/max30
        sma <- as.data.frame(cbind(sma1,sma2,sma3,
                                   sma4,sma5,sma6,sma7))
        names(sma) <- paste(c("SMA2","SMA3","SMA5","SMA10",
                              "SMA15","SMA20","SMA30"))
        return(sma)
    }
    sma <- sma30(price)
    
    # MFI: Money Flow Index
    MFI.f <- function(data,n){
        volume <- as.matrix(data[,"Volume"])
        HLC <- apply(data[,c("High","Low","Close")],1,mean)
        mf <- HLC * volume                    # money flow
        priceLag <- lag(HLC)
        pmf <- ifelse(HLC > priceLag, mf, 0)  # positive money flow
        nmf <- ifelse(HLC < priceLag, mf, 0)  # negative money flow
        mr <- runSum(pmf,n)/runSum(nmf,n)     # money ratio
        mfi <- 100 - (100/(1 + mr))
    }
    mfi30 <- function(data) {
        mfi1 <- MFI.f(data, n=2)
        mfi2 <- MFI.f(data, n=3)
        mfi3 <- MFI.f(data, n=5)
        mfi4 <- MFI.f(data, n=10)
        mfi5 <- MFI.f(data, n=15)
        mfi6 <- MFI.f(data, n=20)
        mfi7 <- MFI.f(data, n=30)
        mfi <- as.data.frame(cbind(mfi1,mfi2,mfi3,mfi4,mfi5,mfi6,mfi7))
        names(mfi) <- paste(c("MFI14", "MFI5", "MFI3", "MFI10",
                              "MFI20", "MFI25", "MFI30"))
        return(mfi)
    }
    mfi <- mfi30(data2)
    
    # WPR: Williams %R (Percent Range)
    wpr.f <- function(data,n){
        high <- data[, "High"]
        low <- data[, "Low"]
        close <- data[, "Close"]
        hmax <- runMax(high, n)
        lmin <- runMin(low, n)
        pctR <- (hmax - close)/(hmax - lmin)
    }
    wpr30 <- function(data) {
        wpr1 <- wpr.f(data, n=2)
        wpr2 <- wpr.f(data, n=3)
        wpr3 <- wpr.f(data, n=5)
        wpr4 <- wpr.f(data, n=10)
        wpr5 <- wpr.f(data, n=15)
        wpr6 <- wpr.f(data, n=20)
        wpr7 <- wpr.f(data, n=30)
        wpr <- cbind(wpr1,wpr2,wpr3,wpr4,wpr5,wpr6,wpr7)
        names(wpr) <- paste(c('wpr1','wpr2','wpr3','wpr4',
                              'wpr5','wpr6','wpr7'))
        return(wpr)
    }
    wpr <- wpr30(data2)
    
    # KD: Stochastic Oscillator
    KD.f <- function(data) {
        high <- data[, "High"]
        low <- data[, "Low"]
        close <- data[, "Close"]
        hmax <- runMax(high, 5)
        lmin <- runMin(low, 5)
        K1 <- (close-lmin)/(hmax-lmin)*100
        D1 <- runMean(K1,3)
        K2 <- lag(K1)
        K3 <- lag(K2)
        D2 <- lag(D1)
        D3 <- lag(D2)
        KD <- cbind(K1,D1,K2,D2,K3,D3)
        names(KD) <- paste(c("K1","D1","K2","D2","K3","D3"))
        return(KD)  
    }
    KD <- KD.f(data2)
    
    # Prediction of PCA
    X.pca <- rffit$X.pca
    X.test <- predict(X.pca, newdata=test[,-1:-2])
    X.test[X.test == "-Inf"] <- NA
    X.test[X.test == "Inf"] <- NA
    X.test1 <- na.omit(X.test)
    
    # Predictions of Random Forest Model
    fit <- rffit$fit
    res.f <- predict(fit, newdata = X.test1)
    
    # Combine to test and output
    Pred <- merge(test, data.frame(res.f), by = 'row.names', all = FALSE)[,c(1,2,3,47)]
    names(Pred)[1] <- "Date"
    names(Pred)[2] <- "Open"
    names(Pred)[3] <- "Close"
    names(Pred)[4] <- "Residual"
    Pred.d <- cbind(stock_select, Pred[nrow(Pred),])
    return(Pred.d)
}



stock_select <- "1478.hk"
data2 <- read_excel(paste(stock_select, "data2.xlsx", sep=""))
test <- read_excel(paste(stock_select, "test.xlsx", sep=""))
load("HK1478.Rdata")
rffit <- HK1478
P1 <- RFpred(stock_select, rffit)


