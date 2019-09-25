
"""
    Created on Tue May 15 23:21:21 2018
    
    @author: junshuaizhang
    """
import arff
import scipy
import scipy.spatial
import numpy as np
import random
import matplotlib.pyplot as plt

def load_autos(path = 'autos.arff.txt'):
    dataset = arff.load(open('autos.arff.txt', 'r'))
    autos = np.array(dataset['data'])
    autos_np_dropped = []
    for row in autos:
        is_ok=1
        for col in row:
            if col is None:
                is_ok = 0
            if col == "?":
                is_ok = 0
        if is_ok == 1:
            autos_np_dropped.append(row)
    autos_np_dropped = np.array(autos_np_dropped)
    autos_X_conti = np.array(autos_np_dropped[:,[0,8,9,10,11,12,15,17,18,19,20,21,22,23,25]], dtype = np.float32)
    autos_Y = np.array(autos_np_dropped[:,24], dtype = np.float32)
    autos_X_nomi = autos_np_dropped[:,[1,2,3,4,5,6,7,13,14,16]]
    return autos_np_dropped,autos_X_conti,autos_X_nomi,autos_Y


# load ionosphere data set
#f1,meta = scipy.io.arff.loadarff("ionosphere.arff.txt")
def load_ionos(path = "ionosphere.arff.txt"):
    dataset = arff.load(open("ionosphere.arff.txt", 'r'))
    ionos_np = np.array(dataset['data'])
    
    
    ionos_X = np.array(ionos_np[:,0:-1], dtype = np.float32)
    map_dict= {'g':1,'b':-1}
    ionos_Y = np.array([ map_dict[ele] for ele in ionos_np[:,-1]], dtype = np.int32)
    return ionos_X,ionos_Y



#input two arrays, one only contains numberical columns, and the other one contains only nominal columns
# will first one-hot encode the nominal array , then merge it with the other numberical array, then output.
def encoding_combining_nominal_cols(conti_array, nomi_array):
    encoded_arrays = []
    for i in range(nomi_array.shape[1]):
        input_col = nomi_array[:,i]
        
        col_set = list(set(input_col.tolist()))
        new_cols = []
        for i, ele in enumerate(input_col):
            new_col = [0]*len(col_set)
            idx = col_set.index(ele)
            new_col[idx] = 1
            new_cols.append(new_col)
        new_cols = np.array(new_cols,dtype=np.int32)
#        print(new_cols)
        encoded_arrays.append(new_cols)
    
    autos_X_nomi_encoded = encoded_arrays[0]
    for i in range(2,len(encoded_arrays)):
        autos_X_nomi_encoded = np.concatenate((autos_X_nomi_encoded, encoded_arrays[i]), axis=1)
    
    
    
    autos_X = np.concatenate((conti_array, autos_X_nomi_encoded), axis=1)
    return autos_X



#binary knn classifier with labels 1 and -1
class KNeighbourClassifier():
    def __init__(self,k=3, weights = "uniform"):
        self.X = None
        self.y = None
        self.k = k
        self.weights = weights
    
    def fit_model(self, X, y):
        self.X = X
        self.y = y
    
    def predict_input(self, X):
        labels = []
        for i in range(len(X)):
            x = X[i]
            count_dict ={1:0,-1:0}
            neigs = self.get_KNeighbours(x)
#            print(neigs)
            for ele in neigs:
                l = self.y[ele[0]]
                if ele[1]==0:
                    w = 1
                else:
                    w = float(1/(ele[1]))
                if self.weights == "distance":
                    count_dict[l]+=1*w
                else:
                    count_dict[l]+=1
            dl = list(count_dict.items())
            dl.sort(key = lambda x:x[1], reverse=True)
            pred = dl[0][0]
            labels.append(pred)
    
        return labels
    
    def get_KNeighbours(self,x):
        distances = []
        for i in range(len(self.X)):
            instance = self.X[i]
            eu_dis = scipy.spatial.distance.euclidean(x,instance)
            distances.append([i,eu_dis])
            distances.sort(key = lambda x:x[1])
        return distances[:self.k]
    
    def evaluate_score(self, X,y):
        labels = self.predict_input(X)
        correct = 0
        for i, l in enumerate(labels):
            l_t = y[i]
            if l==l_t:
                correct+=1
        return float(correct/len(y))




# leave one out cross validation
def loo_cross_validation(clf, X,y):
    preds = []
    evaluate_scores = []
    len_data = len(X)
    ids = random.sample(range(len_data), len_data)
    indices = np.arange(len_data)
    for i in ids:
        test_x = X[i]
        test_y = y[i]
        train_X = X[indices != i, :]
        train_y = y[indices != i]
        clf.fit_model(train_X,train_y)
        pred = clf.predict_input([test_x])
        evaluate_score = clf.evaluate_score([test_x],[test_y])
        preds.append([pred[0],test_y])
        evaluate_scores.append(evaluate_score)
    avg_evaluate_score = np.average(evaluate_scores)
    return avg_evaluate_score


#for weighted knn http://www.data-machine.com/nmtutorial/distanceweightedknnalgorithm.htm
class KNeighbourRegressor():
    def __init__(self,k=3, weights = "uniform"):
        self.X = None
        self.y = None
        self.k = k
        self.weights = weights
    
    def fit_model(self, X, y):
        
        self.X = X
        self.y = y
    def predict_input(self, X):
        preds = []
        for i in range(len(X)):
            x = X[i]
            values = []
            neigs = self.get_KNeighbours(x)
            
            nomis = []
            denomis = []
            for ele in neigs:
                value = self.y[ele[0]]
                if ele[1]==0:
                    w = 1
                else:
                    w = float(1/(ele[1]))
                nomis.append(value*w)
                denomis.append(w)
                values.append(value)
            if self.weights == "distance":
                pred = float(sum(nomis)/sum(denomis))
            else:
                pred = np.average(values)
            
            preds.append(pred)
        
        return preds
    
    def get_KNeighbours(self,x):
        distances = []
        for i in range(len(self.X)):
            instance = self.X[i]
            eu_dis = scipy.spatial.distance.euclidean(x,instance)
            distances.append([i,eu_dis])
            distances.sort(key = lambda x:x[1])
        return distances[:self.k]
    
    
    # Referred to the description on the score function of knr in sklearn
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor.evaluate_score
    # Returns the coefficient of determination R^2 of the prediction
    def evaluate_score_2(self, X,y):
        preds = self.predict_input(X)
        residuals = 0
        mean_value_t = np.average(y)
        sum_of_square = 0
        for i, value in enumerate(preds):
            value_t = y[i]
            residual = (value_t-value)**2
            residuals += residual
            square = (value_t-mean_value_t)**2
            sum_of_square+=square
#        print(sum_of_square)
#        print(residuals)
        if residuals == 0:
            return 1
        if sum_of_square==0:
            return 0
        score = float(1-float(residuals/sum_of_square))
        return score
    
    # mean squared error
    def evaluate_score(self, X,y):
        preds = self.predict_input(X)
        residuals = 0
        #        mean_value_t = np.average(y)
        #        sum_of_square = 0
        for i, value in enumerate(preds):
            value_t = y[i]
            residual = (value_t-value)**2
            residuals += residual
        #            square = (value_t-mean_value_t)**2
        #            sum_of_square+=square
        #        print(sum_of_square)
        #        print(residuals)
        #        if sum_of_square==0:
        #            return 0
        evaluate_score = float(residuals/len(y))
        return evaluate_score



def plot_performance(data_name,data_X,data_y,k_start = 3,k_end = 4):

    x = []
    y_d = []
    y_u = []
    
    for k in range(k_start, k_end+1):
        x.append(k)
        if data_name=="ionosphere":
            clf_d = KNeighbourClassifier(k, weights ="distance")
            clf_u = KNeighbourClassifier(k, weights ="uniform")
        else:
            clf_d = KNeighbourRegressor(k, weights ="distance")
            clf_u = KNeighbourRegressor(k, weights ="uniform")
        p_d = loo_cross_validation(clf_d,data_X,data_y)
        p_u = loo_cross_validation(clf_u,data_X,data_y)
        y_d.append(p_d)
        y_u.append(p_u)
#    print(p_d)
    plt.plot(x, y_u, label="weights: uniform")
    plt.plot(x, y_d, label="weights: distance")
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('Number of Neighbours (k)')
    if data_name=="ionosphere":
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Mean Squared Error')
    plt.title('Performance of varying K values and Weights')
    plt.savefig("{}_{}_{}.png".format(data_name,k_start,k_end))
    plt.show()
    plt.close()
    

