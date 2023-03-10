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
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:  Yale University
 *
 * =====================================================================================
 */

#include<vector>

namespace cudaDPM{
  class Tissue3D{
    public:
      std::vector<int> CellIdx; //vector to indicate the verticies that correspond to which cell
      std::vector<cudaDPM::Cell3D> Cells; //List of cells
      float phi0; //Preffered packing fraction
      int NCELLS;
      int VertDOF;
      int TriDOF;
      float Kc;
      float L;
      float U;

      Tissue3D(std::vector<cudaDPM::Cell3D> cells, float phi0);
      void EulerUpdate(int nsteps, float dt);
      void disperse3D();
      void disperse2D();
  };
  class Tissue2D{
    public:
      std::vector<cudaDPM::Cell2D> Cells;
      float phi0;
      int NCELLS;
      int VertDOF;
      float Kc;
      float L;
      float U;
      int MaxNV;

      Tissue2D(std::vector<cudaDPM::Cell2D> cells, float phi0);
      void EulerUpdate(int nsteps, float dt);
      void disperse();
  };
}
