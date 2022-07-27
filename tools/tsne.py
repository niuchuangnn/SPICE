from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './')
from moco.cifar import CIFAR10


file_features = 'results/cifar10/embedding_res18_cls/feas_moco_512_l2.npy'
file_reliable_labels = 'results/cifar10/eval_res18_cls/labels_reliable.npy'

dataset = CIFAR10("./datasets/cifar10", all=False, transform=None)

labels = dataset.targets

features = np.load(file_features)
reliable_labels = np.load(file_reliable_labels)
idx_select = reliable_labels >= 0

tsne = TSNE(n_components=2, random_state=2000, metric="cosine")

tsne_results1 = tsne.fit_transform(features)

df_subset = {}
df_subset['tsne-2d-one1'] = tsne_results1[:, 0]
df_subset['tsne-2d-two1'] = tsne_results1[:, 1]
df_subset['tsne-2d-one2'] = tsne_results1[idx_select, 0]
df_subset['tsne-2d-two2'] = tsne_results1[idx_select, 1]
df_subset['y1'] = labels
df_subset['y2'] = list(np.array(labels)[idx_select])
plt.figure(figsize=(10, 10))
sns.scatterplot(
    x="tsne-2d-one1", y="tsne-2d-two1",
    hue="y1",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend=None,
    alpha=0.3,
    s=80,
)
plt.axis('off')
plt.savefig('lc_all.png')

plt.figure(figsize=(16, 16))
sns.scatterplot(
    x="tsne-2d-one2", y="tsne-2d-two2",
    hue="y2",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend=None,
    alpha=0.3,
    s=200,
)
plt.axis('off')
plt.savefig('lc_select.png')

plt.show()