import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

f = h5py.File('C:\\Users\\fkampf\\brain_hdf\\neurons_data\\2021-06-16_19-26-46_neurons_data.h5', 'r')

ca_array = f["0"]["cellpose_segmentation"]["F"]

temp_df = pd.DataFrame(ca_array)
temp_df["neuron_name"] = None

for no,item in temp_df.iterrows():
    temp_df.loc[no,"neuron_name"] = "neuron_" + str(no)


features = list(range(1,1971)) # Separating out the features
x = temp_df.loc[:, features].values# Separating out the target
y = temp_df.loc[:,['neuron_name']].values# Standardizing the features
x = StandardScaler().fit_transform(x)


def pca3d(x):
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2','principal component 3'])

    final_df = pd.concat([principalDf, temp_df[['neuron_name']]], axis = 1)

    col1 = final_df.iloc[:,0]
    col2 = final_df.iloc[:,1]
    col3 = final_df.iloc[:,2]

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(col1, col2, col3)
    plt.show()

def pca2d(x):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])

    final_df = pd.concat([principalDf, temp_df[['neuron_name']]], axis = 1)

    col1 = final_df.iloc[:,0]
    col2 = final_df.iloc[:,1]

    plt.scatter(col1, col2)
    plt.show()

pca2d(x)
pca3d(x)



