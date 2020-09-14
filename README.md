**Problem Statement**:
The spread of COVID-19 in the whole world has put the humanity at risk. The resources of
some of the largest economies are stressed out due to the large infectivity and
transmissibility of this disease. Due to the growing magnitude of number of cases and its
subsequent stress on the administration and health professionals, some prediction methods
would be required to predict the number of cases in future.
When will the coronavirus pandemic come to an end? The question is on everyone’s mind,
and while astrologers and politicians have answers, few scientists want to be drawn into
hazarding a prediction.
The challenge of the Covid-19 prediction is the most crucial component for countries and
global health institutions. A successful and accurate prediction to the future covid cases
ultimately results in better management of the pandemic.

Part -01:
The objective of the first part of the problem statement is to predict the Covid Cases of a
City on 1st September 2020. The output file 01 should contain only City and the respective
Covid Cases for the test data.


Part -02:
The Foreign Visitors of a city is a time-dependent parameter, for which you have to come up
with a Time-series prediction model. Using the Foreign Visitors predicted by the model, you
need to calculate the Covid Cases on 1st Oct 2020 for every City in the test data. . The
output file 02 should contain only City and the respective Covid Cases on 1st October.

**Our approach here**
We used a Voting Regression model consisting of three different models: Random forest, Gradient Boosting Regression and Kernel Ridge Regression
After some initial analysis of the data, when we visualized the data, there was no linear trend between different features and target value .So we initially decided to try some tree based models. We started with the basic decision tree but the RMSE value was not that good. Then we used random forest and the RMSE improved drastically. Then we decided to try out few different ensemble learning techniques. We tried XGBoost , Gradient Boosting , Adaboost regression. Among these, gradient boosting helped in improving the RMSE value. We also tried out Kernel Ridge Regression which also improved the RMSE slightly. Then, we decided to merge all these good models in a voting regression model for the final prediction.
Random Forest: Random Forest is a model which consists of many decision trees.
Gradient Boosting regression: Gradient Boosting produces predictive model from an ensemble of weak predictive models.
Kernel ridge regression: It combines Ridge Regression with the kernel trick.
Voting regression: A voting regressor is an ensemble meta-estimator that fits several base regressors, each on the whole dataset. Then it averages the individual predictions to form a final prediction.

For second part,we were supposed to predict covid cases on 1st october using time series prediction model. After analyzing the trend of the time series in Excel using sparklines, there was a clear upward increasing trend in the data. So, we tried Holtz Linear trend model as it is known for good forecasting on time series having a common trend. But because of the NaN values, we were not able to get good forecast and there were many outliers. So, we decided to go with Simple Exponential Smoothing which is based on the principle that the most recent value is attached higher weight than the values from distant past. With that we got a better forecasting. We used a alpha value of 0.9 in the formula of simple exponential smoothing so as to avoid forecasting large values which don’t follow the trend.
