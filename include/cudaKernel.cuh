#include "Cell2D.hpp"
#include "Cell3D.hpp"
#ifndef __KERNEL__
#define __KERNEL__
__global__ void cuShapeForce2D(float dt,int MaxNV, int NCELLS,cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts);
__global__ void cuRetractingForce2D(float dt, int MaxNV, float Kc, float L, int NCELLS, cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts);
__global__ void cuShapeForce3D(int NCELLS,cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Triangles, int offset);
__global__ void cuRepellingForce3D(int NCELLS,bool PBC, float L, float Kc, cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Triangles, int offset);
__device__ void cuUpdateCellVolumes(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset);
__device__ void cuUpdateCOMS(int NCELLS, cudaDPM::Cell3D *Cells,cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset);
__device__ void cuVolumeForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset);
__device__ void cuSurfaceAreaForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces,int offset);
__device__ void cuSurfaceAdhesionUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset);
__global__ void cuSimpleSpringAttraction(int NCELLs,bool PBC, float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts,int offset);
__global__ void cuCatchBondAttraction(int NCELLs,bool PBC, float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts,int offset);
__global__ void cuSlipBondAttraction(int NCELLs,bool PBC, float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, int offset);
__global__ void cuResetForces3D(int NCELLS, int NV, cudaDPM::Vertex3D* Verts, int offset);
__global__ void cuEulerUpdate3D(float dt, int NCELLS, int NV, cudaDPM::Vertex3D *Verts, int offset);
#endif