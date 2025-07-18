from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from progressbar import progressbar

def plot(filename,L,r):
    points = pd.read_csv(filename)
    if r == 2:
        faces = np.uint16(np.genfromtxt('r2tri'))
    elif r == 3:
        faces = np.uint16(np.genfromtxt('r3tri'))
    length = int(len(points.iloc[1:]))
    plt.figure(figsize=(16,16))
    np.random.seed(0)
    r1 = np.random.rand(length)
    r2 = np.random.rand(length)
    r3 = np.random.rand(length)
    try:
        for i in range(0,length,3):
            x = np.mod(points.iloc[i-1,2:],L)
            y = np.mod(points.iloc[i,2:],L)
            plt.scatter(x,y,color=(r1[i],r2[i],r3[i]))
            for tri in faces:
                    vx,vy = [x.iloc[j] for j in tri], [y.iloc[j] for j in tri]
                    if max(vx) - min(vx) <= L/2 and max(vy) - min(vy) <= L/2:
                        plt.plot(vx,vy,color=(r1[i],r2[i],r3[i]))
    except IndexError:
       print("Index Error. Select another recusion number.")

    plt.axis('equal')
    plt.savefig(filename+".png")
    plt.close()

def main():
    args = sys.argv
    if len(args) != 4:
        print("Usage: plottext.py Boxlength isoRecursion Directory")
        return
    print(args[3]);
    files = sorted(glob(args[3]+"*.csv"))
    num_cores = int(multiprocessing.cpu_count())-1
    for i in progressbar(range(0,len(files),num_cores)):
        chunk = files[i:i+num_cores]
        bound_func = partial(plot, L=float(args[1]), r = int(args[2]))
        with ProcessPoolExecutor(max_workers=num_cores) as excutor:
            excutor.map(bound_func,[str(f) for f in chunk])

if __name__ == "__main__":
    main()
