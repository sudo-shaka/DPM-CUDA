
import os
import cudaDPM
from progressbar import progressbar

def main():
  Cell=cudaDPM.Cell3D(x0=7.0,y0=6.0,z0=0,CalA0=1.05,VertexRecursion=2,r0=2.0,Kv=5,Ka=1.5)
  Cell.Ks = 2.0;
  Cell.idealForce = 0.5
  Tissue = cudaDPM.Tissue3D([Cell]*50,0.7)
  Tissue.disperse2D()
  Tissue.Kre = 10.0;
  Tissue.Kat = 0.0;
  Tissue.setAttractionMethod("SimpleSpring")

  nout = 100
  print(Tissue.BoxLength)
  for i in progressbar(range(nout)):
    Tissue.EulerUpdate(200,0.005);
    if not os.path.isdir("output/"):
      os.mkdir("output/")
    filename = './output/'+str(i)+'.csv'
    Tissue.ExportPositions(filename)
if __name__ == "__main__":
  main()
