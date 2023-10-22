# Predicting Stock Market Index and Volatility by AutoML

### VIX daily and weekly signal prediction

CBOE implied volatility index (VIX) daily return signal prediction based on economic and finance variables. 278 comprehensive economic variables are used by a AutoML framework with multiple classification models, including Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, Adaptive Boosting, Multi-Layer Perceptron, and an Ensemble model that combined all methods. 

#### AutoML framework architecture  
![image](https://github.com/yfgit2012/Predicting-Stock-Market-Index-and-Volatility-by-AutoML/assets/5380211/da704782-4f8b-4f8d-a5ff-63f1cdd0e70d)

#### Training and validation accuracy for predicting VIX in 2010-2020

![image](https://github.com/yfgit2012/Predicting-Stock-Market-Index-and-Volatility-by-AutoML/assets/5380211/6c999fa4-aadb-4b0f-bcfe-27a476414c9a)

#### Feature importance analysis   

SHAP Summary Plot          |  SHAP Dependence Plot
:-------------------------:|:-------------------------:
![image](https://github.com/yfgit2012/Predicting-Stock-Market-Index-and-Volatility-by-AutoML/assets/5380211/78220ee1-042b-48ef-af1d-d1ff010e10d4) | ![image](https://github.com/yfgit2012/Predicting-Stock-Market-Index-and-Volatility-by-AutoML/assets/5380211/b8a2235d-dbd1-4395-8927-1bdb4e87d29a)


#### Simulated long-short strategy performance    
The mean daily return in the 11 years between 2010 and 2020. The return is calculated by applying the predicted signal to the next day's VIX return. The diamond is the mean return.    

![image](https://github.com/yfgit2012/Predicting-Stock-Market-Index-and-Volatility-by-AutoML/assets/5380211/68d8a849-2485-431a-a909-52406da3e54e)

We wrote [research paper](https://ssrn.com/abstract=3866415) and [white paper](https://github.com/yfgit2012/ML-fintech-repo/blob/main/VIX%20signal%20prediction%20with%20AutoML/VIX%20Daily%20Directional%20Prediction%20-%20White%20Paper%20V2.3.pdf), feedback and comments are welcome. <br>
 
### SPX and VIX daily return prediction 

Daily prediction of SPX return and VIX level based on the time series and economic and finance variables. Multiple ML algorithms are used with AutoML to optimize the performance. <br><br>
This method can be used for other index funcs, ETFs (SPY, VOO, QQQ, ...), and single stock (AAPL, MSFT, TSLA, GOOGL, ...) level/return prediction. 

### Recent ML based FinTech research literature ([Link](https://github.com/yfgit2012/ML-fintech-repo/tree/main/Literature))
