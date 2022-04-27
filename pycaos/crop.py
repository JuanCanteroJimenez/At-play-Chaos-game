import tensorflow as tf
import numpy as np 



       
def create_points(iter,coef):
    j0 = abs(coef[0]*coef[3]-coef[1]*coef[2])
    j1 = abs(coef[6]*coef[9]-coef[7]*coef[8])
    p = j0/(j0+j1)
    x = 0.05
    y = 0.05
    puntos = []
    for i in range(iter):
        h = np.random.uniform(0,1,1)
        if h > p:
            r = 6
        else:
            r = 0
        x = coef[r + 0]*x + coef[r + 1]*y + coef[4 + r]
        y = coef[r + 2]*x + coef[r + 3]*y + coef[5 + r]
        puntos.append([x,y])
            
                
            
        
    puntos = np.array(puntos[100:len(puntos)-1])
        

class coef_to_array(tf.keras.layers.Layer):
    def __init__(self, length = 256, width = 256, iter = 10000):
        super(coef_to_array, self).__init__()
        
        
    
    @tf.function
    def call(self, inputs):
        def create_points(coef, iterr = iter):
            j0 = abs(coef[0]*coef[3]-coef[1]*coef[2])
            j1 = abs(coef[6]*coef[9]-coef[7]*coef[8])
            p = j0/(j0+j1)
            x = 0.05
            y = 0.05
            puntos = tf.constant([[x,y]])
            for i in range(iterr):
                h = tf.random.uniform(shape=[1], minval=0, maxval=1)
                if h > p:
                    r = 6
                else:
                    r = 0
                xn = coef[r + 0]*x + coef[r + 1]*y + coef[4 + r]
                yn = coef[r + 2]*x + coef[r + 3]*y + coef[5 + r]
                puntos = tf.stack([puntos, tf.constant([[xn, yn]])])
                x = xn
                y = yn
                    
                
            
        
            return(puntos)
        
        points = tf.map_fn(fn = lambda t: create_points(coef=t, iterr = 100), elems=inputs, parallel_iterations=1000)
        return(points)
        
if __name__ == "__main__":
    layer = coef_to_array()
    elems = tf.constant([[0.6, 0.2, -1.0, -0.8, 0.9, -0.7, 0.1, -0.4, 0.2, -0.3, 0.5, 0.1],[-0.5, 0.4, 0.3, 0.7, 0.5, -0.6, 0.2, -0.7, -0.3, -0.8, 0.2, -0.4]],dtype=tf.float64)
    print(elems)
    res = layer(elems)
    print(res)

        
