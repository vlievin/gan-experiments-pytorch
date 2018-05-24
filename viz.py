import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import pickle
import scipy.signal
from collections import defaultdict
import scipy.signal

colors = ['#F44336' , "#00BCD4" , "#FFC107" , "#9C27B0"]

class XY(object):
    def __init__(self):
        self.x = []
        self.y = []
        
class updatable_display2():
    """
    display a graph which is updated every iteration
    """
    def __init__(self, categories=[], headers=[], n_limit = 5000):
        self.categories = categories
        self.headers = headers
        self.n_limit = n_limit
        d = dict()
        for c in categories:
            d[c] = dict()
            for h in headers:
                d[c][h] = XY()
            self.ys = d
            
    def decimate(self, n_max, n_last):
        for c in self.categories:
            h = self.headers[0]
            ratio = len(self.ys[c][h].x) // n_max
            if ratio  > 1:
                for h in self.headers:
                    signal = np.array([self.ys[c][h].x, self.ys[c][h].y])
                    signal = np.concatenate( [signal[:,:-n_last], scipy.signal.decimate(signal[:,-n_last:] , ratio) ] , axis=1)
                    self.ys[c][h].x = list(signal[0])
                    self.ys[c][h].y = list(signal[1])
                    # sort
                    ziped = zip(self.ys[c][h].x , self.ys[c][h].y)
                    ziped = sorted(ziped , key=lambda x:x[0])
                    self.ys[c][h].x = [x[0] for x in ziped]
                    self.ys[c][h].y = [x[1] for x in ziped]


    def update(self, x, c, ys):
        """
        add value
        """
        for h in self.ys[c].keys():
            if type(ys[h]) == type(list):
                self.ys[c][h].y += ys[h]

            else:
                self.ys[c][h].y.append(ys[h])
            if type(x) == type(list):
                self.ys[c][h].x += x
            else:
                self.ys[c][h].x.append(x)
                
        
    def display(self,keys= None,live = True, scale = False):
        """
        plot the training data
        """
        if keys == None:
            keys = self.headers
        plt.clf()
        fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * 5,5), squeeze=False)  
        counter = 0
        for fig_j,c in enumerate(self.categories):
            for fig_i, h in enumerate(keys):
                ax1 = axes[0,fig_i]
                ax1.plot(self.ys[c][h].x,self.ys[c][h].y,colors[fig_j])
                if scale:
                    m = np.nanpercentile(self.ys[c][h].y , 25, interpolation="higher")
                    M = np.nanpercentile(self.ys[c][h].y , 75, interpolation="higher")
                    ax1.set_ylim([0 , 1.5 * M])
                val = self.ys[c][h].y[-1]
                #ax1.set_title(h + ": " +  str(val))
                if counter == 0:
                    ax1.set_title("{0} : {1:.3f}".format(h,val))
                #ax1.annotate(self.ys[h][-1],xy=(   , np.mean(self.ys[h]) ) )
            counter += 1
        fig.tight_layout()
        if live:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.close()
        else:
            plt.plot()
            plt.show()
        
    def close(self):
        """
        close the display
        """
        display.clear_output()
        
    def save(self, filename):
        """
        save data to file
        """
        save = dict()
        save["ys"] = self.ys
        save["categories"] = self.categories
        save["headers"] = self.headers
        pickle.dump( save, open( filename, "wb" ) )
        
    def load(self,filename):
        "load class from file"
        save = pickle.load( open( filename, "rb" ) )
        self.ys = save["ys"]
        self.categories = save["categories"]
        self.headers = save["headers"]
        