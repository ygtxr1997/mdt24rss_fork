import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class TSNEHelper:
    def __init__(self, features: np.ndarray):
        self.features = features
        self.tsne = TSNE(n_components=2, init='pca', random_state=0)
        print('Running T-SNE fit...')
        self.tsne.fit_transform(self.features)
        self.embedding = self.tsne.embedding_
        print(self.embedding.shape)  # (12000, 2)

    def plot_tsne(self):
        fig, ax = plt.subplots(1)
        tsne_result = self.embedding
        y = np.concatenate([
            np.zeros((tsne_result.shape[0] // 2)),
            np.ones((tsne_result.shape[0] // 2))]
        , axis=0).flatten()
        print(tsne_result.shape, y.shape)
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
        lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.savefig('tmp1.png')
        print('fig saved to tmp1.png')
        plt.clf()
        exit()

    def plot_heatmap(self):
        embedding = self.embedding
        min_val = embedding.min()
        max_val = embedding.max()
        heat_map = np.zeros((100, 100), dtype=np.uint8)
        for pairs in embedding:
            px = int((pairs[0] - min_val) / (max_val - min_val) * 98)
            py = int((pairs[1] - min_val) / (max_val - min_val) * 98)
            heat_map[px][py] += 1
        heat_map = ((heat_map / heat_map.max()) * 15).astype(np.uint8)

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.tick_params(labelsize=35)

        import seaborn as sns
        sns.heatmap(heat_map, xticklabels=20, yticklabels=20, cbar=None)

        save_name = 'features_' + self.task_name[:-12] + '.jpg'
        plt.savefig(os.path.join(self.save_path, save_name))
        plt.clf()

        self.heat_map = heat_map