
import cudaDPM
from matplotlib import pyplot as plt
import imageio
from progressbar import progressbar
import time

nv = 30
nc = 100
timesteps=1000
Cell = cudaDPM.Cell2D(0.0,0.0,1.17,nv,1.0,0.05,1.0,0.05)
T = cudaDPM.Tissue2D([Cell]*nc,0.93)
T.disperse()
T.Kc = 1.0;

starttime = time.time()
T.EulerUpdate(timesteps,0.001)
endtime = time.time()

print("It took "+str(float(endtime-starttime))+" seconds to complete "+str(timesteps)+ " timesteps with "+str(nv*nc)+" verticies!")
