from logging import warning
from pickle import TRUE
import numpy as np 
import matplotlib.pyplot as plt
import math 
import matplotlib
import matplotlib.animation as animation
import glob

class ifs_maker:
    def __init__(self) -> None:
        
        
        
        self.coef_voc = np.round(np.arange(-1.2, 1.3, 0.1), 1)
        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H","I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"][0:25]
        self.letter_dic = {self.letters[x]:self.coef_voc[x]  for x in np.arange(len(self.coef_voc))}
        self.bad = False
        self.fd = 0
        self.N1 = 0
        self.N2 = 0
        self.xe = 0.05 + 0.00001
        self.ye = 0.05
        self.lsum = 0
        


    def letters_coef(self, code):
        self.code = [x for x in code]
        self.coef  = [self.letter_dic[x] for x in self.code]
        self.coef2 = [self.letter_dic[x] for x in self.code]
        j0 = abs(self.coef[0]*self.coef[3]-self.coef[1]*self.coef[2])
        j1 = abs(self.coef[6]*self.coef[9]-self.coef[7]*self.coef[8])
        self.p = j0/(j0+j1)
        #self.create_points(self.points,check)
    def create_points(self, iter):
        x = 0.05
        y = 0.05
        self.puntos = []
        self.lsum = 0
        for i in range(iter):
            h = np.random.uniform(0,1,1)
            if h > self.p:
                r = 6
            else:
                r = 0
            x = self.coef[r + 0]*x + self.coef[r + 1]*y + self.coef[4 + r]
            y = self.coef[r + 2]*x + self.coef[r + 3]*y + self.coef[5 + r]
            self.lyapunoc_exponent(i, x, y, r)
            if i >= 100:
                self.puntos.append([x,y])
                self.fractal_dim(i, x, y)
        self.puntos = np.array(self.puntos[100:len(self.puntos)-1])
        

    def initiate_coef(self):
        while(True):
            self.code = np.random.choice(self.letters, 12, replace=True)
            self.coef = [self.letter_dic[x] for x in self.code]
            j0 = abs(self.coef[0]*self.coef[3]-self.coef[1]*self.coef[2])
            j1 = abs(self.coef[6]*self.coef[9]-self.coef[7]*self.coef[8])
            self.p = j0/(j0+j1)
            if (j0 + j1 == 0 or j0 > 1 or j1 > 1):
                pass
            else:
                break

    def d2max_update(self, iter, x, y):
        if iter == 100:
            self.xmax = np.max(np.array(self.puntos)[:, 0])
            self.ymax = np.max(np.array(self.puntos)[:, 1])
            self.xmin = np.min(np.array(self.puntos)[:, 0])
            self.ymin = np.min(np.array(self.puntos)[:, 1])
            self.d2max = (self.xmax- self.xmin)**2 + (self.ymax-self.ymin)**2
        if iter > 100 and iter < 200:
            if x > self.xmax:
                self.xmax = x
            if x < self.xmin:
                self.xmin = x
            if y > self.ymax:
                self.ymax = y
            if y < self.ymin:
                self.ymin = y
            self.d2max = (self.xmax- self.xmin)**2 + (self.ymax-self.ymin)**2
    def xs_ys_update(self, iter):
        if iter == 100:
            
            self.xs = self.puntos[len(self.puntos)-1][0]
            self.ys = self.puntos[len(self.puntos)-1][1]
        if np.random.uniform(0,1,1) < 0.02:
            n = np.random.randint(0, len(self.puntos), 1,dtype=int)
            self.xs = self.puntos[n[0]][0]
            self.ys = self.puntos[n[0]][1]

    def fractal_dim(self, iter, x, y):
        self.d2max_update(iter, x, y)
        self.xs_ys_update(iter)
        dx =  x-self.xs
        dy =  y-self.ys
        d2 = dx*dx + dy*dy
        if d2 < self.d2max*0.001:
            self.N2 += 1
        if d2 < self.d2max*0.00001:
            self.N1 += 1
        if self.N2 != 0 and self.N1 != 0:
            self.fd = 0.434294*np.log(self.N2/self.N1)
    
    def lyapunoc_exponent(self, iter, x, y, r):
        
        xnew = self.coef[r + 0]*self.xe + self.coef[r + 1]*self.ye + self.coef[4 + r]
        ynew = self.coef[r + 2]*self.xe + self.coef[r + 3]*self.ye + self.coef[5 + r]
        dlx = xnew-x
        dly = ynew-y
        dl2 = dlx*dlx + dly*dly
        
        df = dl2
        
        rs = 1/np.sqrt(df)
        self.xe = x + rs*(xnew-x)
        self.ye = y + rs*(ynew-y)
        self.lsum +=  np.log(df)
        self.L = 0.721347*self.lsum/(iter+1)
    def partial_derivate_lyapunoc(self, coef, delta=0.000000001):
        
        self.create_points(5000)
        
        Lprev = self.L
        
        self.coef2[coef] = self.coef2[coef]+delta
        self.create_points(5000)
        Lpost = self.L
        self.coef2 = self.coef.copy()
        
        return((Lpost-Lprev)/delta)
    def gradient_descent_lyapunoc(self, g_iter=1000, step=0.00000000001):
        for _ in range(g_iter):
            self.coef2 = self.coef.copy()
            gradients = []
            for coef in range(12):
                gradients.append(self.partial_derivate_lyapunoc(coef))
            self.coef = np.array(self.coef) - step*np.array(gradients)
            print(self.coef)
            print(self.L)
            print(self.fd)
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            self.create_points(20000)
            plt.subplots(figsize=(900*px, 900*px))

            plt.axis('off')
            
            plt.scatter(self.puntos[:, 0], self.puntos[:, 1], c="black",alpha=0.1, s = 0.1)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

    def partial_derivate_fd(self, coef, delta=0.00001):
        self.create_points(5000)
        Lprev = self.fd
        self.coef2[coef] = self.coef2[coef]+delta
        self.create_points(5000)
        Lpost = self.fd
        self.coef2 = self.coef.copy()
        return((Lpost-Lprev)/delta)
    def gradient_descent_fd(self, g_iter=1000, step=0.000001):
        for _ in range(g_iter):
            self.coef2 = self.coef.copy()
            gradients = []
            for coef in range(12):
                gradients.append(self.partial_derivate_fd(coef))
            self.coef = np.array(self.coef) - step*np.array(gradients)
            print(self.coef)
            print(self.fd)
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            self.create_points(20000)
            plt.subplots(figsize=(300*px, 300*px))

            plt.axis('off')
            
            plt.scatter(self.puntos[:, 0], self.puntos[:, 1], c="black",alpha=0.1, s = 0.1)
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
    def gradient_descent_lyapunoc_ani(self, g_iter=200, step=0.00000000001):
        fig, ax = plt.subplots(figsize=(14,14))
        ims = []
        for i in range(g_iter):
            self.coef2 = self.coef.copy()
            gradients = []
            for coef in range(12):
                gradients.append(self.partial_derivate_lyapunoc(coef))
            self.coef = np.array(self.coef) - step*np.array(gradients)
            print(self.coef)
            
            print(i)
            self.create_points(20000)
            print(self.L)
            print(self.fd)
            x_max =  np.max(self.puntos[:, 0])
           
            y_max =  np.max(self.puntos[:, 1])
            
            
            
            ax.axis("off")
            im = ax.scatter(self.puntos[:, 0]/x_max, self.puntos[:, 1]/y_max, c="black",alpha=0.5, s = 0.1,animated=True)
            ims.append([im])
            
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=False,
                                repeat_delay=1000)

        # To save the animation, use e.g.
        #
        ani.save("".join(self.code) +".gif")
        #
    def gradient_descent_lyapunoc_ani_continuo(self, g_iter=200, step=0.000000001):
        fig, ax = plt.subplots(figsize=(14,14))
        ims = []
        prev = [200,3000,30000]
        up_down = -1
        for i in range(g_iter):
            self.coef2 = self.coef.copy()
            gradients = []
            for coef in range(12):
                gradients.append(self.partial_derivate_lyapunoc(coef))
            """"""
            
            if i > g_iter//2:
                up_down= 1
            
            self.coef = np.array(self.coef) + up_down*step*np.array(gradients)
            
            self.create_points(20000)
            print(up_down)
            
            print(i)
            print(self.L)
            print(self.fd)
            prev.append(self.L)
            x_max =  np.max(self.puntos[:, 0])
           
            y_max =  np.max(self.puntos[:, 1])
            
            
            
            ax.axis("off")
            im = ax.scatter(self.puntos[:, 0]/x_max, self.puntos[:, 1]/y_max, c="black",alpha=0.5, s = 0.1,animated=True)
            ims.append([im])
            
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=False,
                                repeat_delay=1000)

        # To save the animation, use e.g.
        #
        ani.save("continuo"+"".join(self.code) +".gif")
        #
    def quasicir(self, g_iter=100,code="EIJPMWTRCHHO",const=0.2):
        fig, ax = plt.subplots(figsize=(14,14))
        ims = []
        sig = np.random.choice([x + 1 for x in range(4)],size=12,replace=True)
        print(sig)
        up_down = -1
        for i in range(g_iter):
            
            self.letters_coef(code)
            g_iter = 200
            x = np.array([i for _ in range(12)])
            

            mov = ((np.sin(x*((2*np.pi)/(g_iter/2))+sig)+(1/const)))*const
            print(mov)
            self.coef = np.array(self.coef) * mov
            
            self.create_points(7000)
            
            
            print(i)
            print(self.L)
            print(self.fd)
            
            x_max =  np.max(self.puntos[:, 0])
           
            y_max =  np.max(self.puntos[:, 1])
            
            
            
            ax.axis("off")
            im = ax.scatter(self.puntos[:, 0]/x_max, self.puntos[:, 1]/y_max, c="black",alpha=0.5, s = 0.1,animated=True)
            ims.append([im])
            
        ani = animation.ArtistAnimation(fig, ims, interval=32, blit=False,
                                repeat_delay=1000)

        # To save the animation, use e.g.
        #
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        #ani.save("aleatorias_cir/"+str(const)+"cir"+"".join(self.code) +".mp4")
        #

def main():


    filelist =[[x[12:24] for x in glob.glob("aleatorioas/*.png")][30]]

    print(filelist)
    for i in filelist:
        ifs = ifs_maker()
        ifs.quasicir(code = i, const=0.2)
    """"""
    
if __name__ == "__main__":
    main()