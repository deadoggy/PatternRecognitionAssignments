#coding:utf-8

from matplotlib import pyplot as plt
import json
import sys

dataset = "iris"
x_label = range(2, 10)
plot_name = ["mse"]

#kmeans
with open("%s/../kmeans_%s_out.json"%(sys.path[0], dataset), "r") as kmeans_in:
    kmeans_json = json.load(kmeans_in)

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_figheight(3)
fig.set_figwidth(7)
for i, ax in enumerate([axes]):
    y = kmeans_json[plot_name[i]]
    ax.plot(x_label, y, linewidth=3)
    ax.grid()
    ax.set_title(plot_name[i], fontsize=18)

plt.tight_layout()
plt.savefig("%s/../%s_kmeans.png"%(sys.path[0], dataset))

#hierarchical
with open("%s/../hac_%s_out.json"%(sys.path[0], dataset), "r") as hac_in:
    hac_json = json.load(hac_in)

linkages = ["Sgl", "Cpl", "Avg"]
linkage_name = ["Single","Complete","Average"]
fig, axes = plt.subplots(nrows=3, ncols=1)
fig.set_figheight(9)
fig.set_figwidth(7)
for row in xrange(3):
    for col in xrange(1):
        ax = axes[row]
        linkage = linkages[row]
        y =  hac_json[linkage][plot_name[col]]
        ax.plot(x_label, y, linewidth=3)
        ax.grid()
        ax.set_title("%s_%s"%(linkage_name[row], plot_name[col]), fontsize=18)

plt.tight_layout()
plt.savefig("%s/../%s_hac.png"%(sys.path[0], dataset))