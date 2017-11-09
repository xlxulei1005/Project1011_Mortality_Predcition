options(repos=c(CRAN="http://mirrors.ustc.edu.cn/CRAN/"))
###########################################################
#        Use histogram and scatterplot to detect clusters
############################################################

## put histograms on the diagonal
panel.hist <- function(x, ...)
     {
         usr <- par("usr"); on.exit(par(usr))
         par(usr = c(usr[1:2], 0, 1.5) )
         h <- hist(x, plot = FALSE)
         breaks <- h$breaks; nB <- length(breaks)
         y <- h$counts; y <- y/max(y)
         rect(breaks[-nB], 0, breaks[-1], y, col="cyan", ...)
     }


## put bivariate density estimation on the upper panels,
panel.kernel.density <- function (x, y, col = par("col"), bg = NA, pch = par("pch"), 
    cex = 1, ...) 
{
    points(x, y,pch = pch, col =col, bg = bg, cex = cex)
    ok <- is.finite(x) & is.finite(y)
    if (any(ok)) {
    	xl<-list(x1=x,x2=y)
    	xx<-cbind(x,y)
        d <- bkde2D(xx,bandwidth=sapply(xl,dpik))
     contour(x=d$x1,y=d$x2,d$fhat,add=T,...)  
}
}


library(KernSmooth)

#################################################################
###########
###########           Hierarchical Clustering        
###########
#################################################################

###########################################
###  Cars example 
###########################################

# The mtcars data set is built into R:

help(mtcars)

# We will focus on the variables that are continuous in nature rather than discrete:

cars.data <- mtcars[,c(1,3,4,5,6,7)]

# Standardizing by dividing through by the sample range of each variable

samp.range <- function(x){
myrange <- diff(range(x))
return(myrange)
}
my.ranges <- apply(cars.data,2,samp.range)
cars.std <- sweep(cars.data,2,my.ranges,FUN="/") 

#visualizing the data, check the possible clusters

pairs(cars.std, lower.panel=panel.kernel.density, upper.panel=panel.kernel.density, pch = 24, bg="light blue",diag.panel=panel.hist, cex.labels = 2, font.labels=2)
#There is clear evidence of the presence of several separate groups 



# Getting distance matrix:

dist.cars <- dist(cars.std)

# Single linkage:

cars.single.link <- hclust(dist.cars, method='single')

# Plotting the single linkage dendrogram:

plot(cars.single.link, labels=row.names(cars.data), ylab="Distance")

windows() # opening new window while keeping previous one open

# complete linkage:

cars.complete.link <- hclust(dist.cars, method='complete')

# Plotting the complete linkage dendrogram:

plot(cars.complete.link, labels=row.names(cars.data), ylab="Distance")

windows() # opening new window while keeping previous one open

# Average linkage:

cars.avg.link <- hclust(dist.cars, method='average')

# Plotting the average linkage dendrogram:

plot(cars.avg.link, labels=row.names(cars.data), ylab="Distance")

# Centroid linkage:

cars.ct.link <- hclust(dist.cars, method='centroid')

# Plotting the average linkage dendrogram:

plot(cars.ct.link, labels=row.names(cars.data), ylab="Distance")

#Causes an inversion of the dendrogram


# Average Linkage dendrogram seems to indicate two major clusters, 
# Single Linkage dendrogram may indicate three.

#cut the dendrogram
# Single Linkage Solution:

cut.3 <- cutree(cars.single.link, k=3)
cut.3     # printing the "clustering vector"

#alternatively, printing the clusters in terms of the car names
cars.3.clust <- lapply(1:3, function(nc) row.names(cars.data)[cut.3==nc])  
cars.3.clust   

# Cluster 1 seems to be mostly compact cars, Cluster 2 is sports cars, Cluster 3 is large luxury sedans

############# Visualization of Clusters:
### using dendrogram objects
cars.com.d<-as.dendrogram(cars.complete.link)
plot(cars.com.d)

plot(cars.com.d,type="triangle")


### Via the scatterplot matrix:

pairs(cars.data, panel=function(x,y) text(x,y,cut.3))

# Cluster 1 cars tend to have high mileage, low displacement, low horsepower, low weight.
# Cluster 3 cars tend to have low mileage, high weight.


### load package ape; remember to install it: install.packages('ape')
library(ape)
# plot basic tree
plot(as.phylo(cars.complete.link), cex = 0.9)

# cladogram
plot(as.phylo(cars.complete.link), type = "cladogram", cex = 0.9)

plot(as.phylo(cars.complete.link), type = "unrooted")

#see plot.phylo for details.

### Via a plot of the scores on the first 2 principal components, 
### with the clusters separated by color:

cars.pc <- princomp(cars.data,cor=T)

carnames <- abbreviate(row.names(cars.data))

# Setting up the colors for the 3 clusters on the plot:
my.color.vector <- rep("green", times=nrow(cars.data))
my.color.vector[cut.3==2] <- "blue"
my.color.vector[cut.3==3] <- "red"

# Plotting the PC scores:

par(pty="s")
plot(cars.pc$scores[,1], cars.pc$scores[,2], ylim=range(cars.pc$scores[,1]), 
     xlab="PC 1", ylab="PC 2", type ='n', lwd=2)
text(cars.pc$scores[,1], cars.pc$scores[,2], labels=carnames, cex=0.7, lwd=2,
     col=my.color.vector)

# It's clear that PC1 is more important in separating the clusters than PC2.

#alternatively,
clusplot(cars.data, cut.3,labels=1,lines=0)

#or
plot(cars.single.link,labels=row.names(cars.data), ylab="Distance",main="Single linkage")
rect.hclust(cars.single.link,k=3)

#################################################################
#################################################################
###########
###########           Partitioning Clustering        
###########
#################################################################
#################################################################



##########################
##
##  K-means clustering
##
##########################

# A K-means clustering with k = 3:

# Note that the stability of the result can be improved by increasing the maximum number 
# of iterations and using multiple random starts:

cars.k3 <- kmeans(cars.std, centers=3, iter.max=100, nstart=25)
cars.k3

cars.k3.clust <- lapply(1:3, function(nc) row.names(cars.data)[cars.k3$cluster==nc])  
cars.3.clust  
cars.k3.clust

#note that the class lables are different from the results before
############# Visualization of Clusters:

### Via the scatterplot matrix:

pairs(cars.data, panel=function(x,y) text(x,y,cars.k3$cluster))

# Cluster 1 cars tend to have high mileage, low displacement, low horsepower, low weight.
# Cluster 2 cars tend to have low mileage, high weight.


##########################
##
##  K-medoids clustering
##
##########################


# Consider the cars.data and cars.std data frames we created above.

# Let's cluster the cars into k groups using the K-medoids approach.
#The function "pam" is in the "cluster" package.

 
library(cluster)

# K-medoids directly on the (standardized) data matrix:
cars.kmed.3 <- pam(cars.std, k=3, diss=F)

# Or you can do K-medoids by inputting the distance matrix:
# cars.kmed.3 <- pam(dist.cars, k=3, diss=T)

cars.kmed.3$clustering  # printing the "clustering vector"

cars.kmed.3$silinfo$avg.width  #printing the average silhouette width

### A little function to calculate the average silhouette width
### for a variety of choices of k:

my.k.choices <- 2:8
avg.sil.width <- rep(0, times=length(my.k.choices))
for (ii in (1:length(my.k.choices)) ){
avg.sil.width[ii] <- pam(cars.std, k=my.k.choices[ii])$silinfo$avg.width
}
print( cbind(my.k.choices, avg.sil.width) )

# A LARGE average silhouette width indicates that the observations are properly clustered.

# Maybe k=2 is the best choice of k here?


cars.3.clust <- lapply(1:3, function(nc) row.names(cars.data)[cars.kmed.3$clustering==nc])  
cars.3.clust   # printing the clusters in terms of the car names

# Cluster 1 seems to be mostly compact cars, Cluster 2 is sports cars, Cluster 3 is large luxury sedans

############# Visualization of Clusters:

## Built-in plots available with the pam function:

# The "clusplot":

plot(cars.kmed.3, which.plots=1)

# The clusplot (in the "cluster" library) can actually be used with 
# any clustering partition by entering the data set and the clustering vector, e.g.:


# The "silhouette plot":

plot(cars.kmed.3, which.plots=2)

# This shows which observations are "best clustered."

 


#################################################################
#################################################################
###########
###########           Spectral Clustering  
###########
#################################################################
#################################################################

install.packages("kernlab")
library(kernlab)



##simulated data
x1<-seq(from=0.5,to=4.5,by=0.05)
y1<-c(2.5+sqrt(4-(x1-2.5)^2),2.5-sqrt(4-(x1-2.5)^2))
y1<-y1+rnorm(length(y1),sd=0.1)
plot(c(x1,x1),y1,xlab="x",ylab="y")

x2<-seq(from=1.5,to=3.5,by=0.05)
y2<-c(2.5+sqrt(1-(x2-2.5)^2),2.5-sqrt(1-(x2-2.5)^2))
y2<-y2+rnorm(length(y2),sd=0.1)
points(c(x2,x2),y2,xlab="x",ylab="y",col=2)

dd<-cbind(c(x1,x1,x2,x2),c(y1,y2))

#k-means fails
dd.k<-kmeans(dd,2)
plot(dd,col=dd.k$cluster)

#spectral cluster
sc<-specc(dd,centers=2)
plot(dd,col=sc)


#spirals data
data(spirals)
sc <- specc(spirals, centers=2)
plot(spirals, col=sc)

#k-means fails
dd.k<-kmeans(spirals,2)
plot(spirals,col=dd.k$cluster)


####################################################
####################################################
# 
# Choosing the number of clusters k using the
#   average silhouette width criterion.
#
####################################################
####################################################

# When using pam, the output will give you the average 
#silhouette width (see above code).

# We can also get the average silhouette width when using other algorithms:

### With a hierarchical method (Complete linkage here):

summary(silhouette(cutree(cars.complete.link, k=2), dist.cars))$avg.width
summary(silhouette(cutree(cars.complete.link, k=3), dist.cars))$avg.width
 

plot(cars.complete.link, labels=row.names(cars.data), ylab="Distance")
rect.hclust(cars.complete.link, 3)
### With k-means:

summary(silhouette(kmeans(cars.std, centers=2, iter.max=100, nstart=25)$cluster, dist.cars))$avg.width
summary(silhouette(kmeans(cars.std, centers=3, iter.max=100, nstart=25)$cluster, dist.cars))$avg.width

# In each case, we might choose the value of k associated with the LARGEST average silhouette width.


####################################################
####################################################
# 
#    Cluster Validation
#
####################################################
####################################################

  
##  Gap Statistic                                
 
gap.cars<-clusGap(cars.std,kmeans,B=500,K.max=10)
plot(gap.cars)
out.gap<-gap.cars$Tab
maxSE(out.gap[,3],out.gap[,4],method="globalmax")

maxSE(out.gap[,3],out.gap[,4],method="Tibs2001SEmax")
cars.k2 <- kmeans(cars.std, centers=2, iter.max=100, nstart=25)
cars.k2
 

####################################################
####################################################
## 
##  Plotting the WSS for several choices of k
##
####################################################
####################################################

# This is a recommended method for choosing k in K-means clustering.

# For the cars data, let's consider letting k vary up to 5.

#Enter name of the data matrix to be clustered here:
my.data.matrix <- cars.std  

my.k.choices <- 2:8
n <- length(my.data.matrix[,1])
wss1 <- (n-1)*sum(apply(my.data.matrix,2,var))
wss <- numeric(0)
for(i in my.k.choices) {
  W <- sum(kmeans(my.data.matrix,i)$withinss)
  wss <- c(wss,W)
}
wss <- c(wss1,wss)
plot(c(1,my.k.choices),wss,type='l',xlab='Number of clusters', ylab='Within-groups sum-of-squares', lwd=2)
# For what value of k does the elbow of the plot occur?


#CH index
#http://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
install.packages("clusterCrit")
library(clusterCrit)
ch<-numeric(0)
for(k in my.k.choices)
 {  cl <- kmeans(cars.std, centers=k, iter.max=100, nstart=25)
 	ch<-c(ch,intCriteria(as.matrix(cars.std),cl$cluster,"Calinski_Harabasz")$calinski_harabasz)
 }
plot(my.k.choices,ch,type="l")
#  k=2 is the best choice of k

#Rand index
install.packages("fossil")
library(fossil)
rand.index(cars.k3$clus,cars.k2$clus)
adj.rand.index(cars.k3$clus,cars.k2$clus) 


#or use clusterCrit package
#Czekanowski_Dice(F measure) and Rand index
extCriteria(cars.k3$clus,cars.k2$clus,c("Rand","Czekanowski_Dice"))
 
 