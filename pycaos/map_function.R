points <- 10000

MinX <- -1
MaxX <- 1
nX <- 10024
MinY <- -1
MaxY <- 1
nY <- 10024

M <- matrix(c(rnorm(points), runif(points)), ncol = 2 )
dim(M)        
mins <- cbind(rep(MinX, dim(M)[1]), rep(MinY, dim(M)[1]))
pluss <- cbind(rep(nX, dim(M)[1])/(MaxX-MinX), rep(nY, dim(M)[1])/(MaxY-MinY)  )
plot(round((M-mins)*pluss))

