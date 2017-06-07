import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

'''represents best current model'''

raw_housing_data=pd.read_csv('data/kc_house_data.csv')

plt.plot(raw_housing_data['lat'],raw_housing_data['long'], marker='o', alpha=0.2, linestyle='None')
# plt.savefig('Lat_Long_KC.png')
plt.show()

lat_long=raw_housing_data[['lat','long']]
kmeans=KMeans(n_clusters=12)
kmeans.fit(lat_long)
with_zip_km_df=raw_housing_data.copy().fillna(0)
with_zip_km_df = pd.concat([with_zip_km_df, pd.get_dummies(with_zip_km_df['zipcode'],drop_first=True)], axis=1)
with_zip_km_df['km_cluster']=kmeans.labels_
with_zip_km_df = pd.concat([with_zip_km_df, pd.get_dummies(with_zip_km_df['km_cluster'],drop_first=True)], axis=1)
with_zip_km_df['yr_const_work']=with_zip_km_df[["yr_renovated", "yr_built"]].max(axis=1)
with_zip_km_df['renovated']=with_zip_km_df['yr_renovated']>0
with_zip_km_df.drop(['zipcode', 'lat', 'long', 'id', 'date', 'km_cluster', 'yr_renovated', 'yr_built'], axis=1, inplace=True)
# with_zip_km_df.describe()
# with_zip_km_df=with_zip_km_df[with_zip_km_df.apply(lambda x: np.abs(x-x.mean())/float(x.std())< 4, axis=0).all(axis=1)]
with_zip_km_df['ones']=np.ones(with_zip_km_df.shape[0])
price=with_zip_km_df.pop('price')

lr=LinearRegression()
X=StandardScaler().fit_transform(with_zip_km_df)
X_train,X_test,y_train, y_test= train_test_split(X,price)
scores=cross_val_score(lr,X_train,y_train, cv=5)
print 'cross validation score mean'
print np.mean(scores)
lr.fit(X_train,y_train)
print 'final score'
print lr.score(X_test,y_test)
