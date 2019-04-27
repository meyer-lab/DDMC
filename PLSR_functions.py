import scipy as sp, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import explained_variance_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans


###------------ Scaling Matrices ------------------###
'''
Note that the sklearn PLSRegression function already handles regression 
'''
def zscore_columns(matrix):
    matrix_z = np.zeros((matrix.shape[0],matrix.shape[1]))
    for a in range(matrix.shape[1]):
        column = []
        column = matrix[:,a]
        column_mean = np.mean(column)
        column_std = np.std(column)
        matrix_z[:,a] = np.asmatrix([(column-column_mean)/column_std])
    return matrix_z
    
###------------ Q2Y/R2Y ------------------###
'''
Description
Arguments
Returns
'''

def R2Y_across_components(X,Y,max_comps):
    R2Ys = []
    for b in range(1, max_comps):
        plsr = PLSRegression(n_components = b)
        plsr.fit(X, Y)
        R2Y = plsr.score(X,Y)
        R2Ys.append(R2Y)
    return R2Ys

def Q2Y_across_components(X,Y,max_comps):
    Q2Ys = []
    for b in range(1,max_comps):
        plsr_model = PLSRegression(n_components = b)
        y_pred = cross_val_predict(plsr_model, X, Y, cv=Y.size)
        Q2Ys.append(explained_variance_score(Y, y_pred))
    return Q2Ys

###------------ Fitting PLSR and CV ------------------###

def PLSR(X, Y, nComponents):
    plsr = PLSRegression(n_components = nComponents)
    X_scores, Y_scores = plsr.fit_transform(X,Y)
    PC1_scores, PC2_scores = X_scores[:,0], X_scores[:,1]
    PC1_xload, PC2_xload = plsr.x_loadings_[:,0], plsr.x_loadings_[:,1]
    PC1_yload, PC2_yload = plsr.y_loadings_[:,0], plsr.y_loadings_[:,1]
    return plsr, PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload

def MeasuredVsPredicted_LOOCVplot(X,Y,plsr_model, fig, ax, axs): 
    Y_predictions = np.squeeze(cross_val_predict(plsr_model, X, Y, cv=Y.size))
    coeff, pval = sp.stats.pearsonr(list(Y_predictions), list(Y))
    if ax == "none":
        print("Pearson's R: ", coeff, "\n", "p-value: ", pval)
        plt.scatter(Y, np.squeeze(Y_predictions))
        plt.title("Correlation Measured vs Predicted")
        plt.xlabel("Measured Cell Viability")
        plt.ylabel("Predicted Cell Viability")
    else:
        print("Pearson's R: ", coeff, "\n", "p-value: ", pval)
        axs[ax].scatter(Y, np.squeeze(Y_predictions))
        axs[ax].set(title="Correlation Measured vs Predicted",xlabel='Actual Y',ylabel='Predicted Y')  
        
###------------ Phosphopeptide Filter ------------------###

def FilteringOutPeptides(X):            
    NewX = []
    for i, row in enumerate(np.transpose(X)):
        if any(value <= 0.5 or value >= 2 for value in row):
            NewX.append(np.array(list(map(lambda x: np.log(x), row))))
    return np.transpose(np.squeeze(NewX))

###------------ Computing Cluster Averages ------------------###

def ClusterAverages(X_, cluster_assignments, nClusters, nObs):
    X_FCl = np.insert(X_, 0, cluster_assignments, axis = 0)   #11:96   11 = 10cond + clust_assgms
    X_FCl = np.transpose(X_FCl)                        #96:11
    ClusterAvgs = []
    ClusterAvgs_arr = np.zeros((nClusters,nObs))              #5:10
    for i in range(nClusters):
        CurrentCluster = []
        for idx, arr in enumerate(X_FCl):
            if i == arr[0]:                                   #arr[0] is the location of the cluster assignment of the specific peptide
                CurrentCluster.append(arr)                    #array with 96:11, so every arr contains a single peptide's values
        CurrentCluster_T = np.transpose(CurrentCluster)       #11:96, so every arr contains a singlie condition's values (eg. all peptides values within cluster X in Erl)
        CurrentAvgs = []
        for x, arr in enumerate(CurrentCluster_T):
            if x == 0:            #cluster assignments
                continue
            else:
                avg = np.mean(arr)
                CurrentAvgs.append(avg)
        ClusterAvgs_arr[i,:] = CurrentAvgs
        AvgsArr = np.transpose(ClusterAvgs_arr)
    return AvgsArr

###------------ GridSearch ------------------###
'''
Exhaustive search over specified parameter values for an estimator
'''

def GridSearch_nClusters(X):
    kmeans = KMeans(init="k-means++")
    parameters = {'n_clusters': np.arange(2,16)}
    grid = GridSearchCV(kmeans, parameters, cv=X.shape[1])
    fit = grid.fit(np.transpose(X))
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = { '#Clusters': CVresults_max['param_n_clusters'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min


def PipeGridSearch_nClusters(pipe):
    param_grd = dict(kmeans__n_clusters = [1,2,10])
    grid_search = GridSearchCV(pipe, param_grid=param_grid)
    CV_results = grid_search.cv_results_
    best_param = grid_search.best_params_
    return CV_results, best_param

