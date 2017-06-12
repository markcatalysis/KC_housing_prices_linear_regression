# Introduction

The following work was performed as described in order to produce a linear regression model that uses the featurize ddata related to the housing sale in order to first model then predict housing prices in the KC region. The best model performed with an R^2 of 0.813 and incorporated clustering by latitude and longitude of houses.

# Description of Work:

## Goals:

We are building an interpretable model using linear regression to predict housing pricing in KC.  All data except date column are numeric, so it should be a simple application of LinReg model to produce our baseline. Zipcode will need to be dummified as it's one of our neighborhood proxies and not a continuous numeric value. I'm splitting data with a train test split of .75:.25 and holding onto the last quarter of data for final testing.

Note: All scores until the final score will represent the mean across cross validation scores with 5 folds.

For the baseline model, we are dropping zipcode, lat, long, date, yr_renovated, and house ID. I am also filling all NaN with zeros.

The baseline R^2 score for linear regression with was: R^2 = 0.649.

It's time to engineer new features.  Location should play a major role. First I will add dummified zipcodes.  

## First Features to Engineer, Location:
  dummified zipcode   
  clusters by lat_long

With dummified zipcode and dropping the first dummy column, we have increased R^2 to 0.802 which is already a significant improvement and speaks to the importance of location, but there may be more signal contained in location than that stored in zipcode.

I have made a simple x-y plot of the locations by their lat/long values. There is one majority region made of high proximity connected clusters that will be hard to separate. We will first try DBSCAN. If that doesn't provide much additional signal, I will use k-means clustering which imposes explicit "neighborhood" by proximity and should yield more location signal. Both clustering methods will only use the lat and long positions.

Without tuning the DBSCAN hyperparameters and using euclidean distance, we have a new R^2 of 0.803.

With k-means, I chose k=12 clusters as per visual inspection and noting the number of sub-regions that appeared both in the majority-cluster and the outlying regions. R^2 = 0.806, which is a marginal improvement.

Of note, the coefficients corresponding to the number of bedrooms in all these models was negative, which is counterintuitive and highly suspect. We shall attempt to clean up the data and further feature engineering to elucidate if there is misrepresentation of signal, hidden covariance, or other anomalies that would disrupt our model.   

## Adjusting year since most recent construction work performed.
  yrs of most recent work performed on house
    dropping yr_built, yr_renovated
  renovation performed on house

I previously dropped the renovation year largely because the there were zeroes in place of NaNs when a house was never renovated. A feature I am adding in its place will be the year of most recent construction work which will be the max(yr_built and yr_renovated) so as to replace both those columns.

R^2 = 0.803

The score has come down a bit. I will also add a dummified version for whether or not the home was renovated before sale.

R^2 = 0.807

Due to time constraints, I am considering this my final model. The final score on the test data after training on the whole training data set is

R^2 = 0.813

# Conclusion

The model represents a clean interpretable model. Comparing the performance of this model to an untuned *random forest regressor*, we find a similar score of

R^2=0.836

which while higher is harder to directly interpret. The columns which provided the largest linear regression coefficients were square footage of basement and living areas region attenuated (negative coefficients) by number of bedrooms and floors.

# Future Work

## Additional Data Clean Up

I am seeking to remove outliers which may be dramatically pulling our model toward them. Any row that is outside of a zscore of +/- 3 for the data in the column is being removed. Naively removing all rows with any outlier removes all data rows. Doing so will affect how we interpret R^2 as the amount of variance will be scaled to a smaller range. This may seemingly reduce accuracy. All new scores should be interpreted against each other instead of against previous scores.

Also, dummification of columns such as view could improve signal from these largely 0-valued columns.  

## Regularization

I suspect the tests may be overfitting given the number of features I have added so I will use RidgeRegressor to throttle the amount of fitting.  


## Second Features/Changes to Engineer
  date based classification
  date as a numeric
  (price as reevaluated by date?)
  tracking by ID for resales
