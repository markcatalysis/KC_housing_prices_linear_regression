    import numpy as np
import pandas as pd
from scipy import stats
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

raw_housing_data=pd.read_csv('../data/kc_house_data.csv')
raw_housing_data.describe()
# len(raw_housing_data.id.unique())
print raw_housing_data.columns
# raw_housing_data.applymap(np.isreal)

# no location data baseline model preparation. price column will be endogenous variable.

no_loc_df=raw_housing_data.drop(['zipcode', 'lat', 'long', 'id', 'date', 'yr_renovated'], axis=1).fillna(0)
no_loc_df=no_loc_df[no_loc_df.apply(lambda x: np.abs(x-x.mean())/float(x.std())< 3, axis=0).all(axis=1)]
no_loc_df['ones']=np.ones(no_loc_df.shape[0])
# no_loc_df.date=pd.to_datetime(no_loc_df.date)
price=no_loc_df.pop('price')

lr=LinearRegression()
X=StandardScaler().fit_transform(no_loc_df)
X_train,X_test,y_train, y_test= train_test_split(X,price)
X_train,X_test,y_train, y_test= train_test_split(X,price)
scores=cross_val_score(lr,X_train,y_train, cv=5)
np.mean(scores)

# adding zip_code
with_zip_df=raw_housing_data.copy()
with_zip_df = pd.concat([with_zip_df, pd.get_dummies(with_zip_df['zipcode'],drop_first=True)], axis=1)
with_zip_df.drop(['zipcode', 'lat', 'long', 'id', 'date'], axis=1, inplace=True)
with_zip_df=with_zip_df[with_zip_df.apply(lambda x: np.abs(x-x.mean())/float(x.std())< 4, axis=0).all(axis=1)]
with_zip_df['ones']=np.ones(with_zip_df.shape[0])
price=with_zip_df.pop('price')
lr=LinearRegression()
X=StandardScaler().fit_transform(with_zip_df)
X_train,X_test,y_train, y_test= train_test_split(X,price)
scores=cross_val_score(lr,X_train,y_train, cv=5)
np.mean(scores)

# lr.fit(X_train,y_train)
# lr.score(X_test,y_test)

# clustering locations. first plot.

plt.plot(raw_housing_data['lat'],raw_housing_data['long'], marker='.', alpha=0.2, linestyle='None')
plt.savefig('Lat_Long_KC.png')
plt.show()

# DBSCAN clustering

lat_long=raw_housing_data[['lat','long']]
dbscan=DBSCAN()
dbscan.fit(lat_long)
dbscan.labels_
with_zip_db_df=raw_housing_data.copy()
with_zip_db_df = pd.concat([with_zip_db_df, pd.get_dummies(with_zip_db_df['zipcode'],drop_first=True)], axis=1)
with_zip_db_df['db_cluster']=dbscan.labels_
with_zip_db_df = pd.concat([with_zip_db_df, pd.get_dummies(with_zip_db_df['db_cluster'],drop_first=True)], axis=1)
with_zip_db_df.drop(['zipcode', 'lat', 'long', 'id', 'date', 'db_cluster'], axis=1, inplace=True)
with_zip_db_df=with_zip_db_df[with_zip_db_df.apply(lambda x: np.abs(x-x.mean())/float(x.std())< 4, axis=0).all(axis=1)]
with_zip_db_df['ones']=np.ones(with_zip_db_df.shape[0])
price=with_zip_db_df.pop('price')

lr=LinearRegression()
X=StandardScaler().fit_transform(with_zip_db_df)
X_train,X_test,y_train, y_test= train_test_split(X,price)
scores=cross_val_score(lr,X_train,y_train, cv=5)
np.mean(scores)


# kmeans clustering

lat_long=raw_housing_data[['lat','long']]
kmeans=KMeans(n_clusters=12)
kmeans.fit(lat_long)
with_zip_km_df=raw_housing_data.copy()
with_zip_km_df = pd.concat([with_zip_km_df, pd.get_dummies(with_zip_km_df['zipcode'],drop_first=True)], axis=1)
with_zip_km_df['km_cluster']=kmeans.labels_
with_zip_km_df = pd.concat([with_zip_km_df, pd.get_dummies(with_zip_km_df['km_cluster'],drop_first=True)], axis=1)
with_zip_km_df.drop(['zipcode', 'lat', 'long', 'id', 'date', 'km_cluster', 'yr_renovated'], axis=1, inplace=True)
# with_zip_km_df=with_zip_km_df[with_zip_km_df.apply(lambda x: np.abs(x-x.mean())/float(x.std())< 4, axis=0).all(axis=1)]
with_zip_km_df['ones']=np.ones(with_zip_km_df.shape[0])
price=with_zip_km_df.pop('price')

lr=LinearRegression()
X=StandardScaler().fit_transform(with_zip_km_df)
X_train,X_test,y_train, y_test= train_test_split(X,price)
scores=cross_val_score(lr,X_train,y_train, cv=5)
np.mean(scores)
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
# zip(with_zip_km_df.columns,lr.coef_)

# improving on previous model, adding new column of year of last construction/renovation and if renovation performed before sale.

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

# nmf=NMF()
# nmf.fit(X,price)
# zip(with_zip_km_df.columns,nmf.components_[0])
#
# pca=PCA()
# pca.fit(X,price)
# pca.explained_variance_
# zip(with_zip_km_df.columns,pca.components_[0])

X_train,X_test,y_train, y_test= train_test_split(X,price)
scores=cross_val_score(lr,X_train,y_train, cv=5)
np.mean(scores)
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
print zip(with_zip_km_df.columns,lr.coef_)


rf=RandomForestRegressor()
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
