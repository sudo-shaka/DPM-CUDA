
import cudaDPM
from matplotlib import pyplot as plt
from progressbar import progressbar
import matplotlib.pyplot as plt
import imageio
from numpy import random

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
  for ci in range(T.NCELLS):
    for ti in range(Tissue.Cells[ci].NT):
      tri =[T.Cells[ci].FaceIndices[ti].x,T.Cells[ci].FaceIndices[ti].y,T.Cells[ci].FaceIndices[ti].z]
      x = [T.Cells[ci].Verticies[i].x for i in tri]
      y = [T.Cells[ci].Verticies[i].y for i in tri]
      z = [T.Cells[ci].Verticies[i].z for i in tri]
      fx = [T.Cells[ci].Verticies[i].fx for i in tri]
      fy = [T.Cells[ci].Verticies[i].fy for i in tri]
      fz = [T.Cells[ci].Verticies[i].fz for i in tri]
      f = [abs(fx[i])+abs(fy[i])+abs(fz[i]) for i in range(len(fx))]
      ax.plot(x,y,z,color='black')
      #ax.scatter(x,y,z,color=(r1[ci],r2[ci],r3[ci]),animated=True)
      ax.scatter(x,y,z,c=f,cmap='coolwarm',animated=True)
  ax.set_xlim(0,T.BoxLength)
  ax.set_ylim(0,T.BoxLength)
  ax.set_zlim(0,T.BoxLength)

Cell1=cudaDPM.Cell3D(x0=7.0,y0=6.0,z0=1.8,CalA0=1.05,VertexRecursion=2,r0=1.8,Kv=5,Ka=2)
Cell2=cudaDPM.Cell3D(x0=4.0,y0=6.0,z0=2.2,CalA0=1.05,VertexRecursion=2,r0=2.2,Kv=5,Ka=2)
Cell1.Ks = 2.0;
Cell2.Ks = 2.0;
#T = cudaDPM.Tissue3D([Cell1,Cell2],0.5)
T = cudaDPM.Tissue3D([Cell1,Cell2]*16,0.9)
T.disperse2D()
T.Kc = 50.0;

print("Saving data to /tmp/")
nout = 50
with imageio.get_writer('/tmp/out.gif',mode='I') as writer:
  for i in progressbar(range(nout)):
    T.EulerUpdate(50,0.001);
    PlotT3D(T)
    filename = '/tmp/'+str(i)+'.png'
    plt.savefig(filename)
    plt.close()
    image = imageio.imread(filename)
    writer.append_data(image)

