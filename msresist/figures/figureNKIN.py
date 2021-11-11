# plot NKIN Upstream Kinases
plotDistanceToUpstreamKinase(model, [6, 15, 20], ax[5], num_hits=1)
plot_NetPhoresScoreByKinGroup("msresist/data/cluster_analysis/cl6_NKIN.csv", ax[6], n=40, title="Cluster 6 NetworKIN Predictions", color="royalblue")
plot_NetPhoresScoreByKinGroup("msresist/data/cluster_analysis/cl12_NKIN.csv", ax[6], n=40, title="Cluster 12 NetworKIN Predictions", color="royalblue")
plot_NetPhoresScoreByKinGroup("msresist/data/cluster_analysis/cl20_NKIN.csv", ax[7], n=40, title="Cluster 20 NetworKIN Predictions", color="darkorange")