import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

f = h5py.File('C:\\Users\\fkampf\\brain_hdf\\neurons_data\\2021-06-16_19-26-46_neurons_data.h5', 'r')
ca_array = f["0"]["cellpose_segmentation"]["F"]
temp_df = pd.DataFrame(ca_array)
fps = 1/np.nanmean(np.diff(f["0"]['imaging_information'][:, 0]))


def pca3d(unedited_imagingdata,doplot=False):
    temp_df = pd.DataFrame(unedited_imagingdata)
    temp_df["neuron_name"] = None

    for no, item in temp_df.iterrows():
        temp_df.loc[no, "neuron_name"] = "neuron_" + str(no)

    features = list(range(1, len(temp_df.columns) - 1))  # Separating out the features
    x = temp_df.loc[:, features].values  # Separating out the target
    y = temp_df.loc[:, ['neuron_name']].values  # Standardizing the features
    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2','principal component 3'])

    final_df = pd.concat([principalDf, temp_df[['neuron_name']]], axis = 1)

    col1 = final_df.iloc[:,0]
    col2 = final_df.iloc[:,1]
    col3 = final_df.iloc[:,2]
    if doplot ==True:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(col1, col2, col3)
        plt.show()
    return final_df

def pca2d(unedited_imagingdata,doplot=False):

    temp_df = pd.DataFrame(unedited_imagingdata)
    temp_df["neuron_name"] = None

    for no, item in temp_df.iterrows():
        temp_df.loc[no, "neuron_name"] = "neuron_" + str(no)

    features = list(range(1, len(temp_df.columns) - 1))  # Separating out the features
    x = temp_df.loc[:, features].values  # Separating out the target
    y = temp_df.loc[:, ['neuron_name']].values  # Standardizing the features
    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])

    final_df = pd.concat([principalDf, temp_df[['neuron_name']]], axis = 1)

    col1 = final_df.iloc[:,0]
    col2 = final_df.iloc[:,1]
    if doplot==True:
        plt.scatter(col1, col2)
        plt.show()
    return final_df



def pca2d_kmeans(x,doplot=False):
    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=300
    )
    temp_df = pca2d(x)
    kmeans.fit(temp_df.iloc[:,0:2])

    temp_palette = sns.color_palette("tab10",3)
    sns.set_style("darkgrid", {'axes.grid': False})

    if doplot == True:
        ax = sns.relplot(x=temp_df.iloc[:,0], y=temp_df.iloc[:,1], hue=kmeans.labels_, palette=temp_palette,edgecolor="black")

    final_df = pd.concat([temp_df, pd.DataFrame(kmeans.labels_)], axis=1)

    final_df.columns = ['principal component 1', 'principal component 2','neuron_name', "cluster"]
    return final_df




temp_df = pd.concat([temp_df, pca2d_kmeans(ca_array,doplot=True)["cluster"]], axis=1)

average_trace = pd.DataFrame({"0":np.mean(temp_df[temp_df["cluster"]==0]),
                              "1":np.mean(temp_df[temp_df["cluster"]==1]),
                              "2":np.mean(temp_df[temp_df["cluster"]==2])})


#pca2d(x)
#pca3d(x)


