
"""
Created on Tue May 15 23:21:21 2018

@author: junshuaizhang
"""

from modules import modules as mod
from modules.modules import plot_performance
# load autos data set
autos_np_dropped,autos_X_conti,autos_X_nomi,autos_Y = mod.load_autos(path = 'autos.arff.txt')
# convert the nominal data to one-hot encoded arrays and then combine them with other columns with continuous values.
autos_X = mod.encoding_combining_nominal_cols(autos_X_conti, autos_X_nomi)

# load ionos data set
ionos_X,ionos_Y = mod.load_ionos(path = "ionosphere.arff.txt")


# evaluate the ionosphere data using kNN classifier
clf = mod.KNeighbourClassifier(k=1, weights ="uniform")
print("ionosphere data: ",mod.loo_cross_validation(clf,ionos_X,ionos_Y))

# evaluate the autos data using kNN regressor, using only continuous features.
reg = mod.KNeighbourRegressor(k=0, weights ="distance")
print("Mean square error of autos data with continuous features: ",mod.loo_cross_validation(reg,autos_X_conti,autos_Y))
# evaluate the autos data using kNN regressor, using all features including nominal features.
print("Mean square error of autos data with all features: ", mod.loo_cross_validation(reg,autos_X,autos_Y))



#plot_performance(data_name = 'ionosphere',data_X = ionos_X,
#                 data_y = ionos_Y,k_start = 1,k_end =3)

#plot the performance graph for ionosphere data using knn classifier (automatically choose based on data_name)
#plot_performance(data_name = 'ionosphere',data_X = ionos_X,
#                 data_y = ionos_Y,k_start = 1,k_end =3)


##plot the performance graph for ionosphere data using knn classifier (automatically choose based on data_name)
## with all X_data(continuous and nominal)
#plot_performance(data_name = 'autos',data_X = autos_X,
#                 data_y = autos_Y,k_start = 1,k_end = 3)
#
## only continuous
#plot_performance(data_name = 'autos',data_X = autos_X_conti,
#                 data_y = autos_Y,k_start = 1,k_end = 3)
