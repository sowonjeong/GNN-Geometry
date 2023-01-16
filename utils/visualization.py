import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap as UMAP
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from utils.curvature import *
from utils.helper import deg

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def visualize_graph(G, color, size=300):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2", node_size=size)
    plt.show()


def visualize_umap(out, color, size=30, epoch=None, loss = None):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    z = umap_2d.fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()


def visualize_tsne(out, color, size=30, epoch=None, loss=None):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()


def visualize_pca(out, color, size=30, epoch=None, loss=None):
    h = PCA(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(h[:, 0], h[:, 1], s=size, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()


def tight_acc_plot(res, multiple=False, name='plot.png', interval=[0, 1], symm=True):
    data = pd.DataFrame.from_dict(res).T
    data = data[data['gnn_type'] != 'no']
    if symm == False:
        data = data[data['gnn_type'] != 'symmetric diffusion']
    cmap = sb.diverging_palette(220, 20, as_cmap=True)
    if multiple == False:
        data1 = data[data['add_self_loop'] == True]
        fig, axes = plt.subplots(1, 1, figsize=(9, 6), sharex=True, sharey=True)
        xticklabels = ['0',
                       '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
        sb.set_theme(style='whitegrid')
        my_pal = {'normal': 'red', 'diffusion': 'blue', 'symmetric diffusion': 'green'}
        res = sb.boxplot(data=data1,
                         x='alpha',
                         y='test_acc',
                         hue='gnn_type', palette=my_pal)
        res.set_xticklabels(xticklabels, fontsize=25, rotation=0)
        res.set_xlabel(r'$\alpha$', fontsize=30)
        res.set_ylabel('Test Accuracy', fontsize=25)
        res.set_ylim(interval)
        fig.tight_layout(rect=[0, 0, .9, 1])
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        axes.legend(title="Convolution", fontsize=20, title_fontsize=20, loc="lower left")
        new_labels = ["Symmetric", "Row-normalized"]
        for t, l in zip(axes.legend_.texts, new_labels):
            t.set_text(l)
        fig.tight_layout()
        # plt.savefig('plot_experiments_betais1_bis.png')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)
        xticklabels = ['0',
                       '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
        sb.set_theme(style='whitegrid')
        my_pal = {'normal': 'red', 'diffusion': 'blue', 'symmetric diffusion': 'green'}
        res = sb.boxplot(data=data[data['add_self_loop'] == True],
                         x='alpha',
                         y='test_acc',
                         hue='gnn_type', palette=my_pal, ax=axes[0])
        res.set_xticklabels(xticklabels, fontsize=25, rotation=0)
        res.set_xlabel(r'$\alpha$', fontsize=30)
        res.set_ylabel('Test Accuracy', fontsize=25)
        res.set_title("With Self-loop", fontsize=20)

        res2 = sb.boxplot(data=data[data['add_self_loop'] == False],
                          x='alpha',
                          y='test_acc',
                          hue='gnn_type', palette=my_pal, ax=axes[1])
        res2.set_xticklabels(xticklabels, fontsize=25, rotation=0)
        res2.set_xlabel(r'$\alpha$', fontsize=30)
        res2.set_ylabel('', fontsize=25)
        res2.set_title("Without Self-loop", fontsize=20)
        fig.tight_layout(rect=[0, 0, .9, 1])

        axes[0].legend(title="Convolution", fontsize=15, title_fontsize=15, loc="lower left")
        axes[1].legend([], [])
        # axes[1].legend(title="Convolution", fontsize = 15, title_fontsize = 15)
        new_labels = ["Symmetric", "Row-normalized", "Symmetric Row-normalized"]
        for i in range(2):
            for tick in axes[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(25)
            for tick in axes[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(25)
            for t, l in zip(axes[i].legend_.texts, new_labels):
                t.set_text(l)

        fig.tight_layout()
    plt.savefig(name)
    plt.show()


def plot_embedding_transform_PCA(embedding, edge_ind, experiment = 'normal',
alpha = [0.2,0.5,0.8,1.0], name = 'Cora',
trial=1, loop = True, transparency =0.4, mult = 10 ):
    deg1 = deg(edge_ind)+0.01
    # THIS MAKES A GRID and figure size as (width, height) in inches.
    fig, axs = plt.subplots(1, 4, figsize = (23,6))
    sb.set_theme(style="whitegrid")
    #cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(np.log(deg1).min(), np.log(deg1).max())
    title = 'row-normalized' if experiment == 'diffusion' else 'symmetric'
    for i in range(4):
            temp = embedding[name+'_'+experiment+ '_' +str(alpha[i])+'_'+'loop'+'_'+str(loop)+'_'+str(trial)]['embedding']
            tmp = PCA(n_components=2).fit_transform(temp.detach().cpu().numpy())
            _ = axs[i].set_title('PC1 vs PC2: '+ title +', '+ r'$\alpha$' +'=' +str(round(alpha[i],1)), fontsize = 20)
               # _ = axs[j,i].set_xlim([-15, 25])
               # _ = axs[j,i].set_ylim([-10, 35])
            hs = axs[i].scatter(x = tmp[:, 0], y = tmp[:, 1], c=np.log(deg1), s=mult*deg1,  alpha = transparency,  cmap="gist_rainbow")

    cbar = fig.colorbar(hs, ax=axs[3])
    cbar.ax.set_title("log(degree)")

    # Show the graph
    fig.tight_layout()
    plt.show()


def plot_embedding_transform_UMAP(embedding, edge_ind, experiment = 'normal',
alpha = [0.2,0.5,0.8,1.0], name = 'Cora',
trial=1, loop = True, transparency =0.4, mult = 10 ):
    deg1 = deg(edge_ind)+0.01
    # THIS MAKES A GRID and figure size as (width, height) in inches.
    fig, axs = plt.subplots(1, 4, figsize = (23,6))
    sb.set_theme(style="whitegrid")
    #cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(np.log(deg1).min(), np.log(deg1).max())
    title = 'row-normalized' if experiment == 'diffusion' else 'symmetric'
    for i in range(4):
            temp = embedding[name+'_'+experiment+ '_' +str(alpha[i])+'_'+'loop'+'_'+str(loop)+'_'+str(trial)]['embedding']
            tmp = UMAP(n_components=2).fit_transform(temp.detach().cpu().numpy())
            _ = axs[i].set_title('UMAP plot for: '+ title +', '+ r'$\alpha$' +'=' +str(round(alpha[i],1)), fontsize = 20)
               # _ = axs[j,i].set_xlim([-15, 25])
               # _ = axs[j,i].set_ylim([-10, 35])
            hs = axs[i].scatter(x = tmp[:, 0], y = tmp[:, 1], c=np.log(deg1), s=mult*deg1,  alpha = transparency,  cmap="gist_rainbow")

    cbar = fig.colorbar(hs, ax=axs[3])
    cbar.ax.set_title("log(degree)")

    # Show the graph
    fig.tight_layout()
    plt.show()

def plot_embedding_class_PCA(embedding, data, experiment = 'normal',
alpha = [0.2,0.5,0.8,1.0], name = 'Cora', trial=1, loop = True, transparency =0.4, mult = 10, palette="Set2"):
    # out : embedding array
    # deg : data edge_index
    deg1 = deg(data.edge_index)+0.01
    # THIS MAKES A GRID and figure size as (width, height) in inches.
    fig, axs = plt.subplots(1, 4, figsize = (23,6))
    sb.set_theme(style="whitegrid")
    #cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(np.log(deg1).min(), np.log(deg1).max())
    title = 'row-normalized' if experiment == 'diffusion' else 'symmetric'
    for i in range(4):
            temp = embedding[name+'_'+experiment+ '_' +str(alpha[i])+'_'+'loop'+'_'+str(loop)+'_'+str(trial)]['embedding']
            tmp = PCA(n_components=2).fit_transform(temp.detach().cpu().numpy())
            _ = axs[i].set_title('PC1 vs PC2: '+ title +', '+ r'$\alpha$' +'=' +str(round(alpha[i],1)), fontsize = 20)
               # _ = axs[j,i].set_xlim([-15, 25])
               # _ = axs[j,i].set_ylim([-10, 35])
            hs = axs[i].scatter(x = tmp[:, 0], y = tmp[:, 1], c=data.y, s=mult*2,  alpha = transparency,  cmap=palette)
            #_ = axs[i // 5,i % 5].legend(loc="upper right")

    # Show the graph
    savename = name+'_embedding'+'_'+experiment+'_'+'loop'+'_'+str(loop)+'_PCA_class'+'.png'
    fig.tight_layout()
    plt.savefig(savename)
    plt.show()

def plot_embedding_class_UMAP(embedding, data, experiment = 'normal',
alpha = [0.2,0.5,0.8,1.0], name = 'Cora', trial=1, loop = True, transparency =0.4, mult = 10, palette="Set2"):
    # out : embedding array
    # deg : data edge_index
    deg1 = deg(data.edge_index)+0.01
    # THIS MAKES A GRID and figure size as (width, height) in inches.
    fig, axs = plt.subplots(1, 4, figsize = (23,6))
    sb.set_theme(style="whitegrid")
    #cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(np.log(deg1).min(), np.log(deg1).max())
    title = 'row-normalized' if experiment == 'diffusion' else 'symmetric'
    for i in range(4):
            temp = embedding[name+'_'+experiment+ '_' +str(alpha[i])+'_'+'loop'+'_'+str(loop)+'_'+str(trial)]['embedding']
            tmp = UMAP(n_components=2).fit_transform(temp.detach().cpu().numpy())
            _ = axs[i].set_title('UMAP plot for '+ title +', '+ r'$\alpha$' +'=' +str(round(alpha[i],1)), fontsize = 20)
               # _ = axs[j,i].set_xlim([-15, 25])
               # _ = axs[j,i].set_ylim([-10, 35])
            hs = axs[i].scatter(x = tmp[:, 0], y = tmp[:, 1], c=data.y, s=mult*2,  alpha = transparency,  cmap=palette)
            #_ = axs[i // 5,i % 5].legend(loc="upper right")

    # Show the graph
    savename = name+'_embedding'+'_'+experiment+'_'+'loop'+'_'+str(loop)+'_UMAP_class'+'.png'
    fig.tight_layout()
    plt.savefig(savename)
    plt.show()

def plot_embedding_curvature_PCA(embedding, data, experiment = 'normal',
alpha = [0.2,0.5,0.8,1.0], name = 'Cora', trial=1, loop = True, transparency =0.4, mult = 100):
    Adj = nx.adjacency_matrix(to_networkx(data)).todense()
    F_original = augmented_forman(Adj).detach().numpy()
    F_mod = F_original - F_original.min()
    F_mod /= F_mod.max()
    deg1 = deg(data.edge_index)+0.01
    # THIS MAKES A GRID and figure size as (width, height) in inches.
    fig, axs = plt.subplots(1, 4, figsize = (23,6))
    sb.set_theme(style="whitegrid")
    #cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(np.log(deg1).min(), np.log(deg1).max())
    title = 'row-normalized' if experiment == 'diffusion' else 'symmetric'
    for i in range(4):
            temp = embedding[name+'_'+experiment+ '_' +str(alpha[i])+'_'+'loop'+'_'+str(loop)+'_'+str(trial)]['embedding']
            F = augmented_forman(pre_curvature(temp, data))
            tmp = PCA(n_components=2).fit_transform(temp.detach().cpu().numpy())
            axs[i].set_title('PC1 vs PC2: '+ title +', '+ r'$\alpha$' +'=' +str(round(alpha[i],1)), fontsize = 20)

            hs = axs[i].scatter(x = tmp[:, 0], y = tmp[:, 1], c=F, s=mult*F_mod,  alpha = transparency,  cmap="gist_rainbow")


    cbar = fig.colorbar(hs, ax=axs[3])
    cbar.ax.set_title("curvature")

    # Show the graph
    savename = name+'_curvature'+'_'+experiment+'_'+'loop'+'_'+str(loop)+'_PCA'+'.png'
    fig.tight_layout()
    plt.savefig(savename)
    plt.show()
