
fern <- function(vec){
  a <- matrix(c(0,0,
                0,0.16), byrow = TRUE, ncol=2, nrow=2)
  b <- matrix(c(0.85, 0.04,
                -0.04, 0.85), byrow = TRUE, ncol = 2, nrow = 2)
  bb <- c(0,1.60)
  
  j <-  matrix(c(0.20, -0.26,
                 0.23, 0.22), byrow = TRUE, ncol = 2, nrow = 2)
  cc <- c(0,1.60)
  
  d <- matrix(c(-0.15, 0.28,
                0.26, 0.24), byrow = TRUE, ncol = 2, nrow = 2)
  dd <- c(0, 0.44)
  
  m <- list(a,b,j,d)
  p <- list(c(0,0),bb,cc, dd)
  n <- sample(1:4, 1, replace = TRUE, prob = c(0.01, 0.85, 0.07, 0.07))
  
  return(t(m[[n]] %*%  matrix(vec, ncol=1, nrow=2) + p[[n]]))
  
  
}

po <- NULL
xy <- c(-1,0)
for (x in 1:10000){
  xy <- fern(xy)
  po <- rbind(po,xy )
  
  
}
po = cbind(po, 1:nrow(po))
colnames(po) <- c("x", "y", "frame")

library(ggplot2)
library(gganimate)
j <- ggplot(as.data.frame(po)) + aes(x, y) + geom_point(aes(group = seq_along(x)), alpha=0.6, color = "green", shape=".") + theme_classic() + transition_reveal(frame)
j


caos <- function(vec){
  
  a <- matrix(runif(4), byrow = TRUE, ncol=2, nrow=2)
  b <- matrix(runif(4), byrow = TRUE, ncol = 2, nrow = 2)
  bb <- runif(2)
  
  j <-  matrix(runif(4), byrow = TRUE, ncol = 2, nrow = 2)
  cc <-runif(2)
  
  d <- matrix(runif(4), byrow = TRUE, ncol = 2, nrow = 2)
  dd <- runif(2)
  
  m <- list(a,b,j,d)
  p <- list(c(0,0),bb,cc, dd)
  probs <- runif(4)
  probs <- probs/sum(probs)
  n <- sample(1:4, 1, replace = TRUE, prob = probs)
  
  return(t(m[[n]] %*%  matrix(vec, ncol=1, nrow=2) + p[[n]]))
  
  
}

po <- NULL
xy <- c(-1,0)
for (x in 1:10000){
  xy <- caos(xy, 111)
  po <- rbind(po,xy )
  
  
}
po = cbind(po, 1:nrow(po))
colnames(po) <- c("x", "y", "frame")
plot(po[,"x"], po[, "y"])
library(ggplot2)
library(gganimate)
j <- ggplot(as.data.frame(po)) + aes(x, y) + geom_point(aes(group = seq_along(x)), alpha=0.6, color = "green", shape=".") + theme_classic() + transition_reveal(frame)
j






ITS_maker <- function(x){
  while(TRUE){
  coef <- sample(seq(-1.2, 1.2, 0.1), 12, replace = FALSE)
  j0 <- abs(coef[1]*coef[4] - coef[2]*coef[3])
  j1 <- abs(coef[7]*coef[10]- coef[8]*coef[9])
  if(!(j0+j1 == 0 | j0 > 1 | j1 > 1)){break}
  }
  p <- j0/(j0+j1)
  
  function(x,y){
    if(runif(1) > p){n <- 6}else{n <- 0}
    x <-  x*coef[1+n] + y*coef[2+n] + coef[5+n]
    y <-  x*coef[3+n] + y*coef[4+n] + coef[6+n]
    return(c(x,y))
  }
}






library(ggplot2)
set.seed(1)
for (i in 1:5000){
ITS <-ITS_maker()
po <- NULL
xy <- c(0,0)
sto <- TRUE
for (x in 1:20000){
  xy <- ITS(xy[1], xy[2])
  po <- rbind(po,xy )
  if (abs(xy[1]) + abs(xy[2]) > 1000000){sto <- FALSE;break}
}
if(sto){
colnames(po) <- c("x", "y")
scat <- as.data.frame(po[-(1:300),])
p<-ggplot(scat) + geom_point(aes(x, y), alpha = 0.5,shape=20, size = 0.01) + theme(legend.position = "none", panel.grid = element_blank(),axis.title = element_blank(),axis.text = element_blank(),axis.ticks = element_blank(),panel.background =element_blank()) 

ggsave(paste("ITS",i,".png",sep = ""))

}
#ggsave(paste("ITS",i,".png",sep = ""))
}
library(DChaos)
lyapunov(po, m=3:3, lag=1:1, timelapse="FIXED", h=2:10, w0maxit=100,
 wtsmaxit=1e6, pre.white=TRUE, lyapmethod="SLE", blocking="ALL",
 B=100, trace=1, seed.t=TRUE, seed=56666459, doplot=FALSE)






##### Trying to replicate the paper of J. C. Sprott

ITS <-ITS_maker()
po <- NULL
xy <- c(0.05,0.05)
for (x in 1:10000){
  xy <- ITS(xy[1], xy[2])
  po <- rbind(po,xy )
  xsave <- xy[1]
  ysave <- xy[2]
  xy <- ITS(xy[1], xy[2])
  po <- rbind(po,xy )
  xnew <- xy[1]
  ynew <- xy[2]
  
  
  
}
colnames(po) <- c("x", "y")
scat <- as.data.frame(po[-(1:300),])
p<-ggplot(scat) + geom_point(aes(x, y), alpha = 0.5,shape=20, size = 0.01) + theme(legend.position = "none", panel.grid = element_blank(),axis.title = element_blank(),axis.text = element_blank(),axis.ticks = element_blank(),panel.background =element_blank()) 
print(p)






fractal_generator <- function(){
  IFS_maker <- function(x){
    while(TRUE){
      coef <- sample(seq(-1.2, 1.2, 0.1), 12, replace = FALSE)
      j0 <- abs(coef[1]*coef[4] - coef[2]*coef[3])
      j1 <- abs(coef[7]*coef[10]- coef[8]*coef[9])
      if(!(j0+j1 == 0 | j0 > 1 | j1 > 1)){break}
    }
    p <- j0/(j0+j1)
    
    function(x,y,RND){
      if(RND > p){n <- 6}else{n <- 0}
      x <-  x*coef[1+n] + y*coef[2+n] + coef[5+n]
      y <-  x*coef[3+n] + y*coef[4+n] + coef[6+n]
      return(c(x,y))
    }
  }
  
  IFS <- IFS_maker()
  LSUM <- 0
  N1 <- 0
  N2 <- 0
  xy <- c(0.05, 0.05)
  XE <- xy[1] + 0.00001
  YE <- xy[2]
  po <- xy
  for (N in 1:20000){
    if(xy[1] != XE){RND <- runif(1)}
    xy <- IFS(xy[1], xy[2], RND)
    po <- rbind(po, xy)
    if (abs(xy[1]) + abs(xy[2]) > 1000000){sto <- FALSE;break}
    #### Coeficiente de Lyapunov
    XSAVE <- xy[1]
    YSAVE <- xy[2]
    #cat("xy",xy)
    #cat("XE",XE)
    if(xy[1] != XE){RND <- runif(1)}
    xy_new <- IFS(XE, YE, RND)
    
    XNEW <- xy_new[1]
    YNEW <- xy_new[2]
    print(XNEW)
    print(XSAVE)
    DLX <- XNEW - XSAVE
    DLY <- YNEW - YSAVE
    #print(DLX)
    #print(DLY)
    DL2 <- DLX * DLX + DLY * DLY
    DF <- 1e+10 * DL2
    #print(DF)
    RS <- 1/sqrt(DF)
    #print(RS)
    XE <- XSAVE + RS*(XNEW-XSAVE)
    YE <- YSAVE + RS*(YNEW-YSAVE)
    LSUM <- LSUM + log(DF)
    L = 0.721347*LSUM/N
    #print(XE)
    #if(is.na(XE)){break}
    if(N > 1000 & L > -0.2){break}
  }
  #print(L)
  return(po)
}
for (i in 1:1000){
  po <- fractal_generator()
  if (nrow(po) > 10000){
  colnames(po) <- c("x", "y")
  scat <- as.data.frame(po[-(1:300),])
  p<-ggplot(scat) + geom_point(aes(x, y), alpha = 0.5,shape=20, size = 0.01) + theme(legend.position = "none", panel.grid = element_blank(),axis.title = element_blank(),axis.text = element_blank(),axis.ticks = element_blank(),panel.background =element_blank()) 
  print(p)
  Sys.sleep(1)
  }
}

