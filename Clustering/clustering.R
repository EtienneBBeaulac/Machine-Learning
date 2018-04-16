library(datasets)
library(tidyverse)

# Not scaled
dat <- state.x77
plot(hclust(dist(as.matrix(dat))))

# Scaled
dat_scaled <- scale(dat)
plot(hclust(dist(as.matrix(dat_scaled))))

# W/o area
dat_wo_area <- as_tibble(dat_scaled) %>% select(-Area)
plot(hclust(dist(as.matrix(dat_wo_area))))

# Frost only
dat_frost <- as_tibble(dat_scaled) %>% select(Frost)
plot(hclust(dist(as.matrix(dat_frost))))

# All but Frost
dat_wo_frost <- as_tibble(dat_scaled) %>% select(-Frost)
plot(hclust(dist(as.matrix(dat_wo_frost))))


# Cluster into k=5 clusters:
clusters <- kmeans(dat_scaled, 3)

# Summary of the clusters
summary(clusters)

# Centers (mean values) of the clusters
clusters$centers

# Cluster assignments
clusters$cluster

# Within-cluster sum of squares and total sum of squares across clusters
clusters$withinss
clusters$tot.withinss

# Plotting a visual representation of k-means clusters
library(cluster)
clusplot(dat_scaled, clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

tot.withinss <- rep(NULL, 25)

for (k in c(1:25)) {
  clusters <- kmeans(dat_scaled, k)
  tot.withinss[k] <- clusters$tot.withinss
}

plot(tot.withinss)

# Chose 6 as k
clusters <- kmeans(dat_scaled, 6)
summary(clusters)
clusters$centers
clusplot(dat_scaled, clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


# Iris
data(iris)
iris_dat <- as_tibble(iris) %>% 
  select(-Species)
scl_iris_dat <- scale(iris_dat)

for (k in c(1:25)) {
  clusters <- kmeans(scl_iris_dat, k)
  tot.withinss[k] <- clusters$tot.withinss
}
plot(tot.withinss)

plot(hclust(dist(as.matrix(scl_iris_dat))))
clusters <- kmeans(scl_iris_dat, 3)
clusters$cluster
clusplot(scl_iris_dat, clusters$cluster, color=TRUE, shade=TRUE, labels=0, lines=0)
