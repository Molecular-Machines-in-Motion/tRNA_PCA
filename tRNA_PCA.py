# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R


def read_coordinates(path_1, path_2):

    # Reading the prepared coordinate files
    coords_1 = pd.read_csv(path_1, names=['type', 'selection', 'tRNA-elbow-x', 'tRNA-elbow-y', 'tRNA-elbow-z'], delimiter=' ')
    coords_2 = pd.read_csv(path_2, names=['type', 'selection', 'tRNA-CCA-x', 'tRNA-CCA-y', 'tRNA-CCA-z'], delimiter=' ')


    df_all = pd.DataFrame({"state": ["State-1", "State-2-1", "State-2-2", "State-2-3", "State-3-1", "State-3-2",
                                     "State-3-3", "State-3-4", "State-4-1", "State-4-2", "State-5-1", "State-5-2",
                                     "State-5-3"]})
   

    df_all = df_all.join(coords_1['tRNA-elbow-x'])
    df_all = df_all.join(coords_1['tRNA-elbow-y'])
    df_all = df_all.join(coords_1['tRNA-elbow-z'])

    df_all = df_all.join(coords_2['tRNA-CCA-x'])
    df_all = df_all.join(coords_2['tRNA-CCA-y'])
    df_all = df_all.join(coords_2['tRNA-CCA-z'])

    return df_all


if __name__ == '__main__':

    df = read_coordinates('tRNA_C56.txt', 'tRNA_A73.txt')

    # Prepare the data
    to_PCA_tRNA_elbow = df[['tRNA-elbow-x', 'tRNA-elbow-y', 'tRNA-elbow-z']].values
    to_PCA_tRNA_CCA = df[['tRNA-CCA-x', 'tRNA-CCA-y', 'tRNA-CCA-z']].values

    to_PCA_tRNA_both = np.concatenate((to_PCA_tRNA_elbow, to_PCA_tRNA_CCA), axis=0)

    # Running PCA
    PCA_tRNA = PCA(n_components=2)
    PCA_tRNA.fit(to_PCA_tRNA_both)
    PCA_tRNA_transformed = PCA_tRNA.transform(to_PCA_tRNA_both)

    # Preparing dataframe
    df['tRNA_elbow_evx'] = PCA_tRNA_transformed[:int(len(PCA_tRNA_transformed)/2), 1]
    df['tRNA_elbow_evy'] = PCA_tRNA_transformed[:int(len(PCA_tRNA_transformed)/2), 0]

    df['tRNA_CCA_evx'] = PCA_tRNA_transformed[int(len(PCA_tRNA_transformed) / 2):, 1]
    df['tRNA_CCA_evy'] = PCA_tRNA_transformed[int(len(PCA_tRNA_transformed) / 2):, 0]

    # Coloring palette
    palette = {"State-1": '#3E54C0',
               "State-2": "#3E95C0", "State-2-1": "#93B9CB", "State-2-2": "#93B9CB", "State-2-3": "#93B9CB",
               "State-3": '#26FF35', "State-3-1": "#97FFA4", "State-3-2": "#97FFA4", "State-3-3": "#97FFA4", "State-3-4": "#97FFA4",
               "State-4": '#3CC7FF', "State-4-1": "#82E0FF", "State-4-2": "#82E0FF",
               "State-5": "#FFAC14", "State-5-1": "#FFD08C", "State-5-2": "#FFD08C", "State-5-3": "#FFD08C",
               "60S_join": "#FF6258", "Free-tRNA": "#FF81FF"}

    # Plotting
    fig, ax = plt.subplots(figsize=[9, 7])
    sns.scatterplot(x=df.tRNA_CCA_evx, y=df.tRNA_CCA_evy, hue=df.state, palette=palette)
    sns.scatterplot(x=df.tRNA_elbow_evx, y=df.tRNA_elbow_evy, hue=df.state, palette=palette)

    plt.axis('scaled')
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    plt.title('PCA analysis of tRNA dynamics')
    plt.xlabel("PC 1")
    plt.ylabel("PC 0")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('tRNA-CCA-elbow.svg')
    plt.show()

