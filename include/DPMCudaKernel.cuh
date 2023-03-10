/*
 * =====================================================================================
 *
 *       Filename:  cudaDPMCudaKernel.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/02/2022 09:47:27 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
__global__ void cuShapeForce2D(float dt,int MaxNV, int NCELLS,cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts);
__global__ void cuRetractingForce2D(float dt, int MaxNV, float Kc, float L, int NCELLS, cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts);
__global__ void cuShapeForce3D(float dt, int NCELLS,cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Triangles);
__global__ void cuRepellingForce3D(float dt, int NCELLS, int NT, float L, float Kc,
                                   cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D
                                   *Verts, glm::ivec3 *Triangles);
