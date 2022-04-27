library(ggplot2)
library(plotly)

caos2d <- function(iter, point){
 
  atractors <- matrix(runif(point*2, -1,1), ncol = 2, nrow=point)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  while(count < iter){
    n <- sample(1:point,1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}


caos2d_polygon <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
                  
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  while(count < iter){
    n <- sample(1:point,1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}


caos2d_polygon2 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  while(count < iter){
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}


caos2d_polygon3 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  while(count < iter){
    n <- sample((1:point)[-round(n/2 + 1)],1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}


caos2d_polygon4 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  while(count < iter){
    n <- sample((1:point)[-round(n/2 + 2)],1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon5 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  while(count < iter){
    n <- 0 + gamma(n + pi)%%point
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon6 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if ( values[length(values)-2] == values[length(values)-1]){n <- values[length(values)-2]}
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon7 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if ( values[length(values)-2] != values[length(values)-1]){n <- values[length(values)-2]}
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon7 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if ( (values[length(values)-2]) != (values[length(values)-1] )){n <- values[length(values)-2]}
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon8 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if ( 1+(values[length(values)-2]) == (values[length(values)-1] - 1)){n <- values[length(values)-2]}
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon9 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if ( 1+(values[length(values)-2]) == (values[length(values)-1] + 2)){n <- values[length(values)-2]}
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon9 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    n <- 0 + round(sqrt((values[length(values)-1])^2 + (values[length(values)])^2)%%point + 1)
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon10 <- function(iter, point){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:point){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    n <- 1 + round((values[length(values)-1])^2 + (values[length(values)])^2)%%point
    
    n <- sample((1:point)[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon.i <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    
    
    n <- sample((1:(point*2)),1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon.i1 <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    
    
    n <- sample((1:(point*2))[-n],1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon.i2 <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    
    
    n <- sample(c((1:(point*2)), n),1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon.i3 <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if (values[length(values)] == values[length(values)-1]){
    if(n + 1 > point*2){a = 1}else{a = n+1}
    if(n - 1 < 1){b = point*2}else{b = n-1}
    n <- sample((1:(point*2))[-c(a, b)],1, replace = TRUE)
    }else{
      n <- sample((1:(point*2)),1, replace = TRUE)
    }
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon.i4 <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    if (values[length(values)] == values[length(values)-1]){
      if(n + 2 > point*2){a = (n+2)-point*2}else{a = n+2}
      if(n - 2 < 1){b = point*2+(n-2)}else{b = n-2}
      n <- sample((1:(point*2))[-c(a, b)],1, replace = TRUE)
    }else{
      n <- sample((1:(point*2)),1, replace = TRUE)
    }
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}



caos2d_polygon.i.r <- function(iter, point, costant=0.5, rotate=(2*pi)/4){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors_rot <- t(apply(atractors*costant, 1, function(x,y){
    c(x[1]*cos(y)-x[2]*sin(y), x[1]*sin(y) + x[2]*cos(y))
  },y=rotate))
  atractors <- rbind(atractors, atractors_rot)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  while(count < iter){
    
    
    n <- sample((1:(point*2)),1, replace = TRUE)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}


caos2d_polygon.i5 <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  probs<- rbinom(point*2, 10, 0.3)
  probs <- probs/sum(probs)
  while(count < iter){
    
    n <- sample((1:(point*2)),1, replace = TRUE,prob=probs)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}

caos2d_polygon.i6 <- function(iter, point, costant=0.5){
  
  steps <- (2*pi)/point
  grades <- runif(1, 0, 2*pi)
  atractors <- NULL
  for (x in 1:(point)){
    atractors <- rbind(atractors, c(cos(grades), sin(grades)))
    grades <- grades + steps
    
  }
  atractors <- rbind(atractors, atractors*costant)
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  n <- 1
  values <- c(1,1,1)
  probs<- rgeom(point*2, 0.3)
  probs <- probs/sum(probs)
  while(count < iter){
    
    n <- sample((1:(point*2)),1, replace = TRUE,prob=probs)
    
    values <- c(values, n)
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}


caos2d_circle <- function(iter, point){
  
  
  grades <- runif(point, 0, 2*pi)
  atractors <- NULL
  for (x in grades){
    atractors <- rbind(atractors, c(cos(x), sin(x)))
    
    
  }
  print(atractors)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  while(count < iter){
    n <- sample(1:point,1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y")
  result <- as.data.frame(result)
  return(result)
}



set.seed(1)
scat<-caos2d_circle(10000, 5)
ggplot(scat) + geom_point(aes(x, y), alpha = 0.5,shape=20, size = 0.5) + theme(legend.position = "none", panel.grid = element_blank(),axis.title = element_blank(),axis.text = element_blank(),axis.ticks = element_blank(),panel.background =element_blank()) 


caos3d <- function(iter, point){
  
  atractors <- matrix(runif(point*3, -1,1), ncol = 3, nrow=point)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  while(count < iter){
    n <- sample(1:point,1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y", "z")
  result <- as.data.frame(result)
  return(result)
}
j = caos3d(1000, 3)
plot_ly(j, x=~x , y=~y , z=~z)
tetraedro <- j
h = caos3d(1000, 4)
plot_ly(h, x=~x , y=~y , z=~z, type = "scatter3d", mode = "markers", size = 1)
plot(h$x, h$y)
step <- 0.1
for (i in seq(-1,1, step)){
  print(i)
  new <- subset(h, z > i  & z < i+step)
  if (nrow(new) != 0){
  plot(new$x, new$y)
  Sys.sleep(1)}
}








caos4d <- function(iter, point){
  
  atractors <- matrix(runif(point*4, -1,1), ncol = 4, nrow=point)
  init <-  (atractors[1,]+atractors[2,])/2
  count <- 0
  result <- init
  while(count < iter){
    n <- sample(1:point,1, replace = TRUE)
    
    init <- (init+atractors[n,])/2
    result <- rbind(result, init)
    count <- count + 1
  }
  colnames(result) <- c("x", "y", "z", "t")
  result <- as.data.frame(result)
  return(result)
}
j = caos4d(1000, 4)
print(max(j$t))
print(min(j$t))
j <- subset(j,t > -0.9  & t < -0.8)
plot_ly(j, x=~x , y=~y , z=~z)



h = caos4d(500000, 7)
plot_ly(h, x=~x , y=~y , z=~z, color= ~t,type = "scatter3d", mode = "markers", size = 1)

step <- 0.01
pl = 1
l = list()
for (i in seq(-1,1, step)){
  
  new <- subset(h, t > i  & t < i+step)
  if (nrow(new) != 0){
    print("pl")
    l[[pl]] <- new
    pl <- pl + 1
    plot(new$x, new$y)
    title(main = pl)
    Sys.sleep(0.1)
}}

plot_ly(l[[30]], x=~x , y=~y , z=~z, color = ~t, type = "scatter3d", mode = "markers", size = 1)
