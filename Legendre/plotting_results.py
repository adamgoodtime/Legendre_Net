from matplotlib import pyplot as plt
import matplotlib.pylab as pl
# from torch_cluster import grid_cluster, knn_graph
from fast_pytorch_kmeans import KMeans
from torchvision import transforms, datasets
import torch
from tqdm import tqdm
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")

conv_results = torch.load("data/results cluster_all True 10x[10, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000, 40000, 60000] max1000 mnist0.5 sigmoid cnnTrue hidden_size[200] test_acc[99.04].pt")
ff_results = torch.load("data/results cluster_all True 10x[10, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000, 40000, 60000] max1000 mnist0.5 sigmoid hidden_size[200] test_acc[98.1].pt")

# test_labels = [
#     'sigmoid cnnFalse hidden_size[200, 200, 200, 200, 200] test_acc[98.47]',
#     'sigmoid cnnFalse hidden_size[200, 200, 200, 200] test_acc[98.53]',
#     # 'sigmoid cnnFalse hidden_size[400, 400, 400, 400] test_acc[98.54]',
#     'sigmoid cnnFalse hidden_size[200, 200, 200] test_acc[97.9]',
#     'sigmoid cnnFalse hidden_size[200, 200] test_acc[98.11]',
#     # 'relu cnnFalse hidden_size[200, 200] test_acc[98.5]',
#     'sigmoid hidden_size[200] test_acc[98.1]',
#     'sigmoid cnnTrue hidden_size[200] test_acc[99.04]',
#     ]
test_labels = [
    'sigmoid cnnFalse hidden_size[200, 200] test_acc[98.11]',
    'relu cnnFalse hidden_size[200, 200] test_acc[98.5]',
    'sigmoid hidden_size[200] test_acc[98.1]',
    'relu cnnFalse hidden_size[200] test_acc[98.12]',
    '5 relu cnnFalse hidden_size[200] test_acc[98.12]',
    'remove_all True max10',
    'sigmoid cnnTrue hidden_size[200] test_acc[99.04]',
    ]
# test_labels = [
#     'sigmoid cnnFalse hidden_size[1600] test_acc[98.27]',
#     'sigmoid cnnFalse hidden_size[800] test_acc[98.18]',
#     'sigmoid cnnFalse hidden_size[400] test_acc[98.24]',
#     'sigmoid hidden_size[200] test_acc[98.1]',
#     'sigmoid cnnTrue hidden_size[200] test_acc[99.04]',
#     ]
label_labels = {
    'sigmoid cnnFalse hidden_size[400, 400, 400, 400] test_acc[98.54]': '400x4',
    'sigmoid cnnFalse hidden_size[200, 200, 200, 200, 200] test_acc[98.47]': '200x5',
    'sigmoid cnnFalse hidden_size[200, 200, 200, 200] test_acc[98.53]': '200x4',
    'sigmoid cnnFalse hidden_size[200, 200, 200] test_acc[97.9]': '200x3',
    'sigmoid cnnFalse hidden_size[200, 200] test_acc[98.11]': '200x2',
    'sigmoid cnnFalse hidden_size[1600] test_acc[98.27]': '1600x1',
    'sigmoid cnnFalse hidden_size[800] test_acc[98.18]': '800x1',
    'sigmoid cnnFalse hidden_size[400] test_acc[98.24]': '400x1',
    'sigmoid cnnTrue hidden_size[200] test_acc[99.04]': 'CNN5x16-CNN-5x32-200x1',
    'sigmoid hidden_size[200] test_acc[98.1]': '200x1',
    'relu cnnFalse hidden_size[200, 200] test_acc[98.5]': 'ReLU-200x2',
    'relu cnnFalse hidden_size[200] test_acc[98.12]': 'ReLU-200x1',
    '5 relu cnnFalse hidden_size[200] test_acc[98.12]': 'ReLU-200x1-c=5',
    'remove_all True max10': 'random'
}

colours = pl.cm.Set2(np.linspace(0, 1, len(test_labels)))

start_label = "data/results cluster_all True 10x[10, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000, 40000, 60000] max1000 mnist0.5 "

results = {test_label: torch.load(start_label+test_label+'.pt') for test_label in test_labels}
setting = 'vex_cave'
means = {test_label: [{setting: np.mean(results[test_label][k][setting])} for k in results[test_label]]
         for test_label in test_labels}
stds = {test_label: [{setting: np.std(results[test_label][k][setting])} for k in results[test_label]]
         for test_label in test_labels}

k_settings = [k for k in results[test_labels[0]]]
plt.figure()
for idx, test_label in enumerate(means):
    for setting in means[test_label][k_settings[0]]:
        mean = [k[setting] for k in means[test_label]]
        std = [k[setting] for k in stds[test_label]]
        if test_label == 'remove_all True max10':
            mean.append(98.1)
            std.append(0)
        plt.plot(k_settings, mean, label='{}'.format(label_labels[test_label]), color=colours[idx])
        stdev1 = np.array(mean) + np.array(std)
        stdev2 = np.array(mean) - np.array(std)
        plt.fill_between(k_settings, stdev1, stdev2, alpha=0.5, color=colours[idx])
plt.legend(loc="lower right", prop={'size': 17})
plt.xscale('log')
# plt.ylim([-20, 120])
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.ylabel('Test accuracy', fontsize=20)
plt.xlabel('Number of clusters', fontsize=20)
plt.subplots_adjust(left=0.05, bottom=0.1,
                    right=0.995, top=0.993,
                    wspace=0.015, hspace=0)
plt.show()

# conv_ave = [{setting: np.mean(conv_results[k][setting]) for setting in conv_results[k]} for k in conv_results]
# ff_ave = [{setting: np.mean(ff_results[k][setting]) for setting in conv_results[k]} for k in ff_results]
# conv_std = [{setting: np.std(conv_results[k][setting]) for setting in conv_results[k]} for k in conv_results]
# ff_std = [{setting: np.std(ff_results[k][setting]) for setting in conv_results[k]} for k in ff_results]
setting = 'vex_cave'
conv_ave = [{setting: np.mean(conv_results[k][setting])} for k in conv_results]
ff_ave = [{setting: np.mean(ff_results[k][setting])} for k in ff_results]
conv_std = [{setting: np.std(conv_results[k][setting])} for k in conv_results]
ff_std = [{setting: np.std(ff_results[k][setting])} for k in ff_results]

k_settings = [k for k in ff_results]

plt.figure()
for setting in conv_ave[k_settings[0]]:
    mean = [k[setting] for k in conv_ave]
    std = [k[setting] for k in conv_std]
    plt.plot(k_settings, mean, label='c{}'.format(setting))
    stdev1 = np.array(mean) + np.array(std)
    stdev2 = np.array(mean) - np.array(std)
    plt.fill_between(k_settings, stdev1, stdev2, alpha=0.5)
for setting in ff_ave[k_settings[0]]:
    mean = [k[setting] for k in ff_ave]
    std = [k[setting] for k in ff_std]
    plt.plot(k_settings, mean, label='f{}'.format(setting))
    stdev1 = np.array(mean) + np.array(std)
    stdev2 = np.array(mean) - np.array(std)
    plt.fill_between(k_settings, stdev1, stdev2, alpha=0.5)
plt.legend(loc="lower right")
# plt.ylim([-20, 120])
plt.ylabel('Test accuracy')
plt.xlabel('Number of clusters')
plt.show()

print("done")
