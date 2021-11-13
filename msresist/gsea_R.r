library(clusterProfiler)
library(ggplot2)

# Oncology & General Gene Sets
cm <- read.gmt("msresist/data/cluster_analysis/c6.all.v7.4.entrez.gmt")

data <- read.csv("msresist/data/cluster_analysis/CPTAC_GSEA_Input.csv")
on <- compareCluster(Gene~Clusters, data=data, fun=enricher, TERM2GENE=cm, pAdjustMethod='hochberg')
go <- compareCluster(Gene~Clusters, data=data, fun="enrichGO", ont="BP", pAdjustMethod='bonferroni', OrgDb='org.Hs.eg.db')
mf <- compareCluster(Gene~Clusters, data=data, fun="enrichGO", ont="MF", pAdjustMethod='bonferroni', OrgDb='org.Hs.eg.db')
cc <- compareCluster(Gene~Clusters, data=data, fun="enrichGO", ont="CC", pAdjustMethod='bonferroni', OrgDb='org.Hs.eg.db')
wp <- compareCluster(Gene~Clusters, data=data, fun="enrichWP", pAdjustMethod='bonferroni', organism="Homo sapiens")

ggplot(wp, aes(x=reorder(Clusters, sort(as.numeric(Clusters))), y=Description, colour=p.adjust)) + geom_point(aes(size=GeneRatio)) + theme_bw() + ggtitle("") + theme(plot.title = element_text(hjust = 0.5)) + xlab("Clusters") + ylab("Process")

OutputPath = '/Users/creixell/Desktop/UCLA/Projects/AXLinteractors/CPTAC_gsea_WP.svg'
ggsave(OutputPath, units='cm', width=30, height=15)

# Immunological Gene Sets
im <- read.gmt("msresist/data/cluster_analysis/c7.immunesigdb.v7.4.entrez.gmt")
imrF <- compareCluster(Gene~Clusters, data=data, fun=enricher, TERM2GENE=im, pAdjustMethod='hochberg')
bdrF <- compareCluster(Gene~Clusters, data=data, fun=enricher, TERM2GENE=bd, pAdjustMethod='hochberg')

imdata <- read.csv("msresist/data/cluster_analysis/CPTAC_GSEA_ImmuneClusters.csv")
imr <- compareCluster(Gene~Clusters, data=imdata, fun=enricher, TERM2GENE=im, pAdjustMethod='hochberg')
wpr <- compareCluster(Gene~Clusters, data=imdata, fun="enrichWP", pAdjustMethod='bonferroni', organism="Homo sapiens")

ggplot(wpr, aes(x=reorder(Clusters, sort(as.numeric(Clusters))), y=Description, colour=p.adjust)) + geom_point(aes(size=GeneRatio)) + theme_bw() + ggtitle("") + theme(plot.title = element_text(hjust = 0.5)) + xlab("Clusters") + ylab("Process")

OutputPath = '/Users/creixell/Desktop/UCLA/Projects/AXLinteractors/CPTAC_gsea_Immuno_Full.svg'
ggsave(OutputPath, units='cm', width=30, height=40)