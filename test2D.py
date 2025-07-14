
import cudaDPM
from matplotlib import pyplot as plt
import imageio
from progressbar import progressbar


def EulerTest(nout,nsteps, dt, ncells):
  C0 = cudaDPM.Cell2D(0.0,0.0,1.4,30,2.4,0.1,1.0,0.05)
  C1 = cudaDPM.Cell2D(0.0,0.0,1.37,25,2.0,0.1,1.0,0.05)
  T = cudaDPM.Tissue2D([C0,C1]*int(ncells/2),0.95)
  T.disperse();
  T.Kc = 2.0

  #The reset is just plotting...
  print("Saving data to /tmp/")
  with imageio.get_writer('/tmp/cuTest2d.gif',mode='I') as writer:
    for s in progressbar(range(nout)):
      X = []; Y = []; F = []
      filename = '/tmp/' + str(s) + 'cu.png'
      fig = plt.figure(figsize=(15,15))
      for ci in range(T.NCELLS):
        for vi in range(T.Cells[ci].NV):
          x = T.Cells[ci].Verticies[vi].x
          y = T.Cells[ci].Verticies[vi].y
          if x < 0:
            x += T.BoxLength*round(abs(x/T.BoxLength)+1)
          if x > T.BoxLength:
            x -= T.BoxLength*(round(abs(x-T.BoxLength)/T.BoxLength) + 1)
          if y < 0:
            y += T.BoxLength*round(abs(y/T.BoxLength)+1)
          if y > T.BoxLength:
            y -= T.BoxLength*(round(abs(y-T.BoxLength)/T.BoxLength) + 1)
          f = abs(T.Cells[ci].Verticies[vi].fx) + abs(T.Cells[ci].Verticies[vi].fy)
          f *= 0.5
          X.append(x); Y.append(y); F.append(f)
      plt.scatter(X,Y,c=F,cmap='coolwarm')
      plt.axis('equal')
      plt.savefig(filename)
      plt.close()
      image = imageio.imread(filename)
      writer.append_data(image)
      T.EulerUpdate(nsteps,dt)

if __name__ == "__main__":
  EulerTest(100,2500,0.001,150)
