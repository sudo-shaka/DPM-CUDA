 #include "Cell.hpp"
__global__ void cuShapeForce2D(float dt,int MaxNV, int NCELLS,cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts);
__global__ void cuRetractingForce2D(float dt, int MaxNV, float Kc, float L, int NCELLS, cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts);
__global__ void cuShapeForce3D(int NCELLS,cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Triangles);
__global__ void cuRepellingForce3D(int NCELLS,bool PBC, float L, float Kc, cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Triangles);
__device__ void cuUpdateCellVolumes(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces);
__device__ void cuUpdateCOMS(int NCELLS, cudaDPM::Cell3D *Cells,cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces);
__device__ void cuVolumeForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces);
__device__ void cuSurfaceAreaForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces);
__device__ void cuSurfaceAdhesionUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces);
__global__ void cuSimpleSpringAttraction(int NCELLs,bool PBC, float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts);
__global__ void cuCatchBondAttraction(int NCELLs,bool PBC, float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts);
__global__ void cuSlipBondAttraction(int NCELLs,bool PBC, float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts);
__global__ void cuResetForces3D(int NCELLS, int NV, cudaDPM::Vertex3D* Verts);
__global__ void cuEulerUpdate3D(float dt, int NCELLS, int NV, cudaDPM::Vertex3D *Verts);
