from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率 





def use_pac_process_data(x, dim):
    pca=PCA(n_components=dim)
    trans_x=pca.fit_transform(x) 
    return trans_x

def plot_pca_result(pca_data, target, dim,lb2name, title="", savefig="pca.png"):
    x_min, x_max = pca_data[:,0].min() - 0.5, pca_data[:,0].max()+0.5
    y_min, y_max = pca_data[:,1].min() - 0.5, pca_data[:,1].max()+0.5
    color = ['grey', 'gold', 'darkviolet', 'turquoise', 'r', 'g', 'b', 'c', 'm', 'y',
    'k', 'darkorange', 'lightgreen', 'plum', 'tan',
    'khaki', 'pink', 'skyblue', 'lawngreen', 'salmon']
    node_types = len(lb2name)
    types = []
    for i in range(node_types):
        types.append([])
    for i in range(len(target)):
        types[target[i]].append(np.array(pca_data[i]))
    for i in range(node_types):
        types[i] = np.array(types[i])


    # plt.legend(handles=[g1, g2, g3], labels=['not at all', 'a small doses', 'a large doses'])

    if dim==2:
        handles = []
        for i in range(node_types):
            handles.append(plt.scatter(types[i][:,0], types[i][:,1], c = color[i]))
        plt.legend(handles=handles,labels=[lb2name[i] for i in range(node_types)])
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(node_types):
            ax.scatter(types[i][:,0], types[i][:,1], types[i][:,2], c = color[i], alpha=0.8,label=lb2name[i])
        ax.legend(loc='best')
        
    # plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig(savefig)
    plt.show()
    
def pca(x,y,dim,lb2name,save="temp.png"):
    trans_x = use_pac_process_data(x, dim)
    plot_pca_result(trans_x, y, dim, lb2name, title="PCA", savefig=save)

if __name__ == '__main__':
    dataset = "IMDB"
    test_method = "final"
    train_percent = 20
    feature_mode = ""
    ablation = "all"
    path = np.load("pathvec\pathvec" + dataset + "_" + str(
        train_percent) + "_" + test_method + "_" + feature_mode + ablation + ".npy",)
    final_onehot = np.load("pathvec\pathlabel" + dataset + "_" + str(
        train_percent) + "_" + test_method + "_" + feature_mode + ablation + ".npy")
    import pickle

    with open("pathvec\pathname" + dataset + "_" + str(
            train_percent) + "_" + test_method + "_" + feature_mode + ablation + ".npy", "rb") as f:
        onehot2name = pickle.load(f)
    index = np.random.choice(list(range(path.shape[0])), 1000, replace=False)
    path = path[index, :]
    final_onehot = final_onehot[index]
    pca(path, final_onehot, 3, onehot2name)