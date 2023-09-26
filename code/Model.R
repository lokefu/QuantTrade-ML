setwd("~/Desktop/HKU/Academic/Y4S2/Capstone-STAT3799")

library(tseries); library(magrittr); library(dplyr)
library(xts); library(TTR); library(randomForest); library(readxl); library(writexl)



## Benchmark


# benchmark_HK = "^HSI" # Hang Seng Index, standard for HK stocks
# benchmark_US = "^DJUS" # Dow Jones U.S. Index, standard for US stocks
# benchmark_US_Tech = "^GSPC" # S&P 500, standard for US tech stocks
# benchmark_US_Industrial = "^DJI" # Dow Jones Industrial Average, standard for US industrial stocks

# For US market, S&P500 and DJUS provide sector indexes.
    # https://www.spglobal.com/spdji/en/index-family/equity/us-equity/sp-sectors/#indices
    # https://www.spglobal.com/spdji/en/index-family/equity/us-equity/dow-jones-sectors/#indices
    
# For HK market, HSI provides sector indexes.
    # https://www.hsi.com.hk/eng/indexes/all-indexes#sector



## Model


RFmodel <- function(stock_select, relationship = 0) {
    
    
    ## Benchmark stock(s)' return
    
    getRtn <- function(stock_select) {
        library(tseries)
        x <- get.hist.quote(instrument = stock_select, 
                            provider = 'yahoo',
                            quote = c("Open", "High", "Low", 
                                      "Close","Adjusted", "Volume"),
                            start = "2016-01-01")   
        # Remove observations with missing values
        data1 <- data.frame(x)
        data1[data1 == "null"] = NA
        data1[data1 == "0"] = NA
        data1 <- na.omit(data1)
        
        # Calculate daily log returns of open prices in the future
        library(magrittr) #pipe
        price <- data1[,"Open"]
        rtn <- price %>% log %>% diff #log difference of stock price
        # Should use excess returns
        return (rtn)
    }
    
    
    ## Target stock' return
    
    library(tseries)
    # Load data
    x <- get.hist.quote(instrument = stock_select, 
                        provider = 'yahoo',
                        quote = c("Open", "High", "Low", 
                                  "Close","Adjusted", "Volume"),
                        start = "2016-01-01")   
    # Remove observations with missing values
    data1 <- data.frame(x)
    data1[data1 == "null"] = NA
    data1[data1 == "0"] = NA
    data1 <- na.omit(data1)
    
    # Calculate daily log returns of open prices in the future
    library(magrittr) #pipe
    price <- data1[,"Open"]
    rtn <- price %>% log %>% diff #log difference of stock price
    # Should use excess returns
    HSI = data.frame(getRtn("^HSI"))
    rtn1 <- data.frame(rtn)
    
    
    ## Dummy variables & CAPM model
    
    if (relationship == 0){
        # no sign relationship
        lmdata <- data.frame(target = rtn1, market = HSI[0:nrow(rtn1),])
        model <- lm(rtn ~ market, data=lmdata)
        res = data.frame(resid(model))
    }
    if (relationship == 1){
        # + ~ +, target returns always have same sign with market returns
        pos <- ifelse(HSI$getRtn...HSI.. >= 0, 1, 0)
        neg <- ifelse(HSI$getRtn...HSI.. < 0, 1, 0)
        pos = data.frame(pos)
        neg = data.frame(neg)
        lmdata <- data.frame(target = rtn1,
                             market = HSI[0:nrow(rtn1),],
                             positive = pos[0:nrow(rtn1),],
                             negative = neg[0:nrow(rtn1),])
        model <- lm(rtn ~ market + positive + negative, data=lmdata)
        res = data.frame(resid(model)) 
    }
    if (relationship == 2){
        # + ~ -, target returns always have opposite sign with market returns
        pos <- ifelse(HSI$getRtn...HSI.. >= 0, 0, 1)
        neg <- ifelse(HSI$getRtn...HSI.. < 0, 0, 1)
        pos = data.frame(pos)
        neg = data.frame(neg)
        lmdata <- data.frame(target = rtn1,
                             market = HSI[0:nrow(rtn1),],
                             positive = pos[0:nrow(rtn1),],
                             negative = neg[0:nrow(rtn1),])
        model <- lm(rtn ~ market + positive + negative, data=lmdata)
        res = data.frame(resid(model)) 
    }
    
    data2 <- data.frame(cbind(data1[0:nrow(res),], res))
    data2 <- na.omit(data2)
    
    write_xlsx(data2, paste("~/Desktop/HKU/Academic/Y4S2/Capstone-STAT3799/", stock_select, "data2.xlsx", sep=""))
    
    data3 <- data2[c("Open", "Close", "resid.model.")]
    price <- data2$Close
    
    
    ## Explanatory variables/indicators
    
    # RSI: Relative strength Index
    rsi30 <- function(x) {
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
    macd30 <- function(x) {
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
    sma30 <- function(x) {
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
        names(wpr) <- paste(c("WPR2","WPR3","WPR5","WPR10",
                              "WPR15","WPR20","WPR30"))
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
    
    # Add indicators to data4 to generate data5
    data4 <- cbind(data3, rsi, macd, sma, mfi, wpr, KD)
    data5 <- na.omit(data4)
    
    
    ## Train/Test
    
    data5[data5 == "-Inf"] <- NA
    data5[data5 == "Inf"] <- NA
    data6 <- na.omit(data5)
    
    # Use about 5 years' data as train and 1.5 years' data as test
    N <- floor(0.75*nrow(data6))
    train <- data6[0:N, ]
    test <- data6[N:nrow(data6), ]
    test = select(test, -3)
    
    write_xlsx(test, paste("~/Desktop/HKU/Academic/Y4S2/Capstone-STAT3799/", stock_select, "test.xlsx", sep=""))
    
    
    ##############  I. PCA ############## 
    
    X.pca <- prcomp(train[,-1:-3], retx= , center=T, scale.=T)
    # print(X.pca)
    # plot(X.pca, type = "l")
    # summary(X.pca)
    # biplot(X.pca)
    
    # Combine the cluster with the PCA vectors
    train1 <- data.frame(train, X.pca$x)
    train2 <- train1[,-4:-46]
    
    
    ############## II. Random Forest ###############
    
    library(randomForest)
    set.seed(2022)
    train3 <- train2[,-1:-2]
    
    # Random Forest Models
    fit <- randomForest(resid.model. ~ ., 
                        data = train3)
    fits <- list(X.pca, fit)
    names(fits) <- c('X.pca', 'fit')
    
    return(fits)
}



stock_select <- "1478.hk"
HK1478 <- RFmodel(stock_select, 0)
save(HK1478, file = "HK1478.Rdata")


