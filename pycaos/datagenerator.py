from logging import warning
from os import sep
from pickle import TRUE
import numpy as np 
import matplotlib.pyplot as plt
import math 
import matplotlib
import pandas as pd
import time


class ifs_maker:
    def __init__(self, initc, points, check) -> None:
        self.check = check
        self.initc = initc
        self.points = points
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
        self.max_iter = 0
        if self.initc:
            self.initiate_coef()
            self.create_points(self.points, self.check)


    def letters_coef(self, code,check):
        self.code = [x for x in code]
        self.coef = self.coef = [self.letter_dic[x] for x in self.code]
        j0 = abs(self.coef[0]*self.coef[3]-self.coef[1]*self.coef[2])
        j1 = abs(self.coef[6]*self.coef[9]-self.coef[7]*self.coef[8])
        self.p = j0/(j0+j1)
        #self.create_points(self.points,check)
    def create_points(self, iter,check):
        start = time.time()
        x = 0.05
        y = 0.05
        self.puntos = []
        for i in range(iter):
            h = np.random.uniform(0,1,1)
            if h > self.p:
                r = 6
            else:
                r = 0
            if abs(x) + abs(y) > 1000000:
                self.bad = True
                break
            
            
            if i > 1000 and check :
                

                if self.fd < 1 or self.L > -0.2:
                    #print("malo")
                    self.bad = True
                    break
            x = self.coef[r + 0]*x + self.coef[r + 1]*y + self.coef[4 + r]
            y = self.coef[r + 2]*x + self.coef[r + 3]*y + self.coef[5 + r]
            self.lyapunoc_exponent(i, x, y, r)
            if i >= 100:
                self.puntos.append([x,y])
                self.fractal_dim(i, x, y)
                
            
        
        self.puntos = np.array(self.puntos[100:len(self.puntos)-1])
        end = time.time()
        self.max_iter = (end-start)

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
        

class ifs_maker2:
    def __init__(self, initc, points, check, atractors=2) -> None:
        if atractors > 4:
            print("no implementado")
        else:
            self.atractors = atractors
        self.check = check
        self.initc = initc
        self.points = points
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
        self.times = 0
        if self.initc:
            self.initiate_coef()
            self.create_points(self.points, self.check)


    def letters_coef(self, code,check):
        self.code = [x for x in code]
        self.coef = self.coef = [self.letter_dic[x] for x in self.code]
        j0 = abs(self.coef[0]*self.coef[3]-self.coef[1]*self.coef[2])
        j1 = abs(self.coef[6]*self.coef[9]-self.coef[7]*self.coef[8])
        self.p = j0/(j0+j1)
        #self.create_points(self.points,check)
    def create_points(self, iter,check):
        start = time.time()
        x = 0.05
        y = 0.05
        self.puntos = []
        for i in range(iter):
            r = np.random.choice([x*6 for x in range(self.atractors)], size=1)[0]
            """if abs(x) + abs(y) > 1000000:
                self.bad = True
                break"""
            
            
            
            if i > 1000 and check :
                

                if self.fd < 1 or self.L > -0.2:
                    #print("malo")
                    self.bad = True
                    break
            
            x = self.coef[r + 0]*x + self.coef[r + 1]*y + self.coef[4 + r]
            y = self.coef[r + 2]*x + self.coef[r + 3]*y + self.coef[5 + r]
            self.lyapunoc_exponent(i, x, y, r)
            if i >= 100:
                self.puntos.append([x,y])
                self.fractal_dim(i, x, y)
                
            
        
        self.puntos = np.array(self.puntos[100:len(self.puntos)-1])
        end = time.time()
        self.times = i
        

    def initiate_coef(self):
        while(True):
            self.code = np.random.choice(self.letters, (self.atractors-2)*6 + 12, replace=True)
            self.coef = [self.letter_dic[x] for x in self.code]
            j0 = abs(self.coef[0]*self.coef[3]-self.coef[1]*self.coef[2])
            j1 = abs(self.coef[6]*self.coef[9]-self.coef[7]*self.coef[8])
            
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
        


    
    


def main():
    
    
    for x in range(100):
        ifs = ifs_maker(False,20000, True)
        ifs.letters_coef("QFKKGLFCMDXO")
        print(ifs.code)
        print(ifs.letter_dic)

        ifs.coef = ifs.coef #+ np.random.normal(0, 0.1, 12)
        ifs.create_points(40000, True)





        

        plt.axis('off')
        plt.scatter(ifs.puntos[:, 0], ifs.puntos[:, 1],c="black",alpha=1, s = 1)
        plt.show()
        
def main2():
    ifs = ifs_maker(True,2000)
    print(ifs.letter_dic)


def main3():
    j = 0
    while(True):


        ifs = ifs_maker(False, 20000, True)
        ifs.letters_coef("SJQWODPLIHHU", True)
        ifs.coef = ifs.coef * np.random.normal(1,0.1,12)
        ifs.create_points(40000, True)
        
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        if (ifs.bad):
            pass
        else:
            j += 1
            if(j > 10000):
                break

            
            print(j)
            print("L",ifs.L)
            print(ifs.fd)
            plt.subplots(figsize=(1200*px, 1200*px))

            plt.axis('off')
            plt.scatter(ifs.puntos[:, 0], ifs.puntos[:, 1], c="black",alpha=0.1, s = 0.1)
            
            plt.savefig("unique3/"+str(j)+"".join(ifs.code)+".png", dpi=plt.rcParams['figure.dpi'])
            plt.close()


def main7():
    j = 0
    ifs = ifs_maker(False, 20000, True)
    ifs.letters_coef("JNOVKFRXLTGR", True)
    while(True):



        print("ok")
        ifs.coef = ifs.coef + np.array([0,0,0,0,0.01,0,0,0,0,0,0,0])
        ifs.create_points(20000, True)
        
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        if (ifs.bad):
            pass
        else:
            j += 1
            if(j > 10000):
                break

            
            print(j)
            print("L",ifs.L)
            print(ifs.fd)
            plt.subplots(figsize=(1200*px, 1200*px))

            plt.axis('off')
            plt.scatter(ifs.puntos[:, 0], ifs.puntos[:, 1], c="black",alpha=0.9, s = 0.1)
            
            plt.savefig("unique2/"+str(j)+"".join(ifs.code)+".png", dpi=plt.rcParams['figure.dpi'])
            plt.close()



def main5():
    j = 0
    
    while(True):
        ifs = ifs_maker(False, 20000, True)
        let = np.random.choice(["EBPVNNSEKGQJ","SJQWODPLIHHU" ,"ELCJXJVVJGOF", "JMPUSTSKRVNB"])
        ifs.letters_coef(let, True)
        ifs.coef = ifs.coef * np.random.normal(1,0.2,12)
        ifs.create_points(4000, True)

        
        
        
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        if (ifs.bad):
            pass
        else:
            j += 1
            if(j > 10000):
                break

            
            print(j)
            print("L",ifs.L)
            print(ifs.fd)
            plt.subplots(figsize=(64*px, 64*px))

            plt.axis('off')
            plt.scatter(ifs.puntos[:, 0], ifs.puntos[:, 1], c="black",alpha=0.5, s = 0.01)
            
            plt.savefig("caos_images/"+str(j)+"".join(ifs.code)+".png", dpi=plt.rcParams['figure.dpi'])
            plt.close()       

def main9():
    j = 0
    while(True):


        ifs = ifs_maker(False, 20000, True)
        ifs.letters_coef("SJQWODPLIHHU", True)
        ifs.coef = ifs.coef + np.array([0,0,0,0,0,0,0.01,0,0,0,0,0])*j # * np.random.normal(1,0.2,12)
        ifs.create_points(10000, True)
        
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        if (ifs.bad):
            pass
        else:
            j += 1
            if(j > 10000):
                break

            
            print(j)
            print("L",ifs.L)
            print(ifs.fd)
            plt.subplots(figsize=(1200*px, 1200*px))

            plt.axis('off')
            plt.scatter(ifs.puntos[:, 0], ifs.puntos[:, 1], c="black",alpha=0.5, s = 0.1)
            
            plt.savefig("unique3/"+str(j)+"".join(ifs.code)+".png", dpi=plt.rcParams['figure.dpi'])
            plt.close()

def main10():
    j = 0
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    result = []
    for x in range(100000):
        print("x",x)
        for atractor in range(2,5,1):
            ifs = ifs_maker2(True, 5000, False, atractor)
            
            
            
            if ifs.bad == False:
                print(ifs.L)
                print(ifs.fd)

                #plt.subplots(figsize=(1200*px, 1200*px))
                result.append([ifs.L, ifs.fd, atractor, "".join(ifs.code)])
                #plt.axis('off')
                
                #plt.scatter(ifs.puntos[:, 0], ifs.puntos[:, 1], c="black",alpha=0.5, s = 0.1)
                #plt.show(block=False)
                #plt.pause(1)
                #plt.close()
    resultados = pd.DataFrame(result)
    resultados.columns = ["L", "FD", "Atractores", "Id"]
    resultados.to_csv("datos_investigacion.csv", sep=",")
    print(resultados)
    
                
def main11():
    control = 0
    result = []
    while control <= 4000:
        for atractor in range(2,5,1):
            px = 1/plt.rcParams['figure.dpi']
            ifa = ifs_maker2(True, 20000, True, atractor)
            #ifa.letters_coef("JLMUQTGSMXDA", True)
            ifa.create_points(20000, True)
            #plt.subplots(figsize=(1200*px, 1200*px))
            #plt.axis('off')
            
            if ifa.bad==True:
                print("L", ifa.L)
                print("time", ifa.times)
                print("FD", ifa.fd)
                print(control)
                control += 1
                result.append([ifa.L, ifa.fd, atractor, "".join(ifa.code), ifa.times])
                """plt.scatter(ifa.puntos[:, 0], ifa.puntos[:, 1], c="black",alpha=0.5, s = 0.1)
                plt.show(block=False)
                plt.pause(1)
                plt.close()""" 
    resultados = pd.DataFrame(result)
    resultados.columns = ["L", "FD", "Atractores", "Id","time"]
    resultados.to_csv("datos_fallidos_investigacion.csv", sep=",")
    print(resultados)         
            



if __name__ == "__main__":
    main11()


    




