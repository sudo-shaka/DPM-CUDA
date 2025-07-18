/*
 * =====================================================================================
 *
 *       Filename:  Tissue.hpp
 *
 *    Description: Functions for interacting cudaDPM in 2D and 3D
 *
 *        Version:  1.0
 *        Created:  06/02/2022 08:17:43 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Shaka,
 *   Organization:  Yale University
 *
 * =====================================================================================
 */

 #ifndef __TISSUE__
 #define __TISSUE__
#include<vector>
#include"Cell3D.hpp"
#include"Cell2D.hpp"
#include<string>
#include<functional>
#include<unordered_map>

namespace cudaDPM{
  class Tissue3D{
    public:
      std::vector<cudaDPM::Cell3D> Cells; //List of cells
      float phi0; //Preffered packing fraction
      int NCELLS;
      int VertDOF;
      float Kre;
      float Kat;
      float L;
      bool PBC;
      std::unordered_map<std::string, std::function<void(Tissue3D*,int)>> methods{
        {"SimpleSpring",[](Tissue3D *t, int offset){t->SimpleSpringAttraction(offset);}},
        {"CatchBond",[](Tissue3D *t, int offset){t->CatchBondAttraction(offset);}},
        {"SlipBond",[](Tissue3D *t,int offset){t->SlipBondAttraction(offset);}}
      };
      void setAttractionMethod(std::string);
      void SimpleSpringAttraction(int offset);
      void CatchBondAttraction(int offset);
      void SlipBondAttraction(int offset);
      Tissue3D(std::vector<cudaDPM::Cell3D> cells, float phi0);
      void EulerUpdate(int nsteps, float dt);
      void disperse3D();
      void disperse2D();
      void SetUpCudaMemory();
      void ExtractCudaMemory();
      void exportForcesToCSV(std::string filename);
      void exportPositionsToCSV(std::string filename);

    private:
      std::string attMethodName = "SimpleSpring";
      std::function<void(int offset)> attractionMethod;
      cudaDPM::Cell3D* CellsCuda = NULL;
      cudaDPM::Vertex3D* VertsCuda = NULL;
      glm::ivec3* TriCuda = NULL;
  };
  class Tissue2D{
    public:
      std::vector<cudaDPM::Cell2D> Cells;
      float phi0;
      int NCELLS;
      int VertDOF;
      float Kre;
      float Kat;
      float L;
      float U;
      int MaxNV;

      Tissue2D(std::vector<cudaDPM::Cell2D> cells, float phi0);
      void EulerUpdate(int nsteps, float dt);
      void disperse();
  };
}
#endif