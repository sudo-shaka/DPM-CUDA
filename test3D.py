
import cudaDPM
from progressbar import progressbar
import os

def main():
  Cell=cudaDPM.Cell3D(x0=7.0,y0=6.0,z0=0,CalA0=1.05,VertexRecursion=2,r0=2.0,Kv=5,Ka=1.5)
  Cell.Ks = 3.00;
  Cell.idealForce = 1.0
  Tissue = cudaDPM.Tissue3D([Cell]*100,0.65)
  Tissue.disperse2D()
  Tissue.Kre = 10.0;
  Tissue.Kat = 1.0;
  Tissue.setAttractionMethod("SimpleSpring")

  nout = 50
  print(Tissue.BoxLength)
  if not os.path.isdir("output/"):
    os.mkdir("output/")
  for i in progressbar(range(nout)):
     Tissue.EulerUpdate(100,0.005);
     filename = 'output/'+str(i)+'.csv'
     Tissue.ExportPositions(filename)
if __name__ == "__main__":
  main()
