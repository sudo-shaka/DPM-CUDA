
import cudaDPM
from matplotlib import pyplot as plt
from progressbar import progressbar
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from numpy import random
import numpy as np

def PlotCell(Cell):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.view_init(10,120)
  for Vert in Cell.Verticies:
    x = Vert.x
    y = Vert.y
    z = Vert.z
    ax.scatter(x,y,z, color = 'blue',animated=True)

def PlotT3D(Tissue):
  random.seed(10)
  r1 = random.rand(Tissue.NCELLS)
  r2 = random.rand(Tissue.NCELLS)
  r3 = random.rand(Tissue.NCELLS)
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(projection='3d')
  ax.view_init(-90,-180)
  ci = 0;
  for ci in range(Tissue.NCELLS):
    for ti in range(Tissue.Cells[ci].NT):
      tri =[Tissue.Cells[ci].FaceIndices[ti].x,Tissue.Cells[ci].FaceIndices[ti].y,Tissue.Cells[ci].FaceIndices[ti].z]
      x = [Tissue.Cells[ci].Verticies[i].x for i in tri]
      y = [Tissue.Cells[ci].Verticies[i].y for i in tri]
      z = [Tissue.Cells[ci].Verticies[i].z for i in tri]
      fx = [Tissue.Cells[ci].Verticies[i].fx for i in tri]
      fy = [Tissue.Cells[ci].Verticies[i].fy for i in tri]
      fz = [Tissue.Cells[ci].Verticies[i].fz for i in tri]
      f = [abs(fx[i])+abs(fy[i])+abs(fz[i]) for i in range(len(fx))]
      ax.plot(x,y,z,color='black')
      ax.scatter(np.mod(x,Tissue.BoxLength),np.mod(y,Tissue.BoxLength),np.mod(z,Tissue.BoxLength),color=(r1[ci],r2[ci],r3[ci]),animated=True)
      #ax.scatter(np.mod(x,Tissue.BoxLength),np.mod(y,Tissue.BoxLength),z,c=f,cmap='coolwarm',animated=True)
  ax.set_xlim(0,Tissue.BoxLength)
  ax.set_ylim(0,Tissue.BoxLength)
  ax.set_zlim(0,Tissue.BoxLength)

def main():
  Cell=cudaDPM.Cell3D(x0=7.0,y0=6.0,z0=0,CalA0=1.05,VertexRecursion=2,r0=2.0,Kv=5,Ka=1.5)
  Cell.Ks = 5.0;
  Cell.idealForce = 0.5
  Tissue = cudaDPM.Tissue3D([Cell]*8,0.65)
  Tissue.disperse2D()
  Tissue.Kre = 10.0;
  Tissue.Kat = 1.0;
  Tissue.setAttractionMethod("SimpleSpring")

  print("Simulation doesn't take that long. It's plotting that takes forever")
  print("Saving data to /tmp/")
  nout = 100
  print(Tissue.BoxLength)
  with imageio.get_writer('/tmp/out.gif',mode='I') as writer:
    for i in progressbar(range(nout)):
      Tissue.EulerUpdate(200,0.005);
      PlotT3D(Tissue)
      filename = '/tmp/'+str(i)+'.png'
      plt.savefig(filename)
      plt.close()
      image = imageio.imread(filename)
      writer.append_data(image)
if __name__ == "__main__":
  main()
