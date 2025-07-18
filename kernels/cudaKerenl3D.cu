#define GLM_ENAME_EXPERIMENTAL
#include "Cell3D.hpp"
#include "cudaKernel.cuh"
#include<cmath>
#include <glm/geometric.hpp>
#include<glm/glm.hpp>
#include<glm/vec3.hpp>
#include<glm/mat3x3.hpp>

__device__ void cuUpdateCellVolumes(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset){
  int ci = blockIdx.x + offset;
  if(ci >= NCELLS){
    return;
  }
  int fi = threadIdx.x;
  int NV = Cells[ci].NV;

  glm::ivec3 face = Faces[fi];

  int i0 = ci * NV + face.x;
  int i1 = ci * NV + face.y;
  int i2 = ci * NV + face.z;

  glm::vec3 P0 = {Verts[i0].X, Verts[i0].Y, Verts[i0].Z};
  glm::vec3 P1 = {Verts[i1].X, Verts[i1].Y, Verts[i1].Z};
  glm::vec3 P2 = {Verts[i2].X, Verts[i2].Y, Verts[i2].Z};

  glm::mat3 Positions={P0,P1,P2};
  __shared__ float cellVolume;
  if(threadIdx.x == 0) cellVolume = 0.0f;
  __syncthreads();
  float vol = glm::dot(glm::cross(P0,P1),P2)/6.0f;
  atomicAdd(&cellVolume,vol);
    __syncthreads();
  if(threadIdx.x == 0){
    Cells[ci].Volume = fabsf(cellVolume);
  }
}

__device__ void cuUpdateCOMS(int NCELLS, cudaDPM::Cell3D *Cells,cudaDPM::Vertex3D *Verts, int offset){
  int ci = blockIdx.x + offset; 
  int fi = threadIdx.x;
  if(ci >= NCELLS){
    return;
  }
  int NV = Cells[ci].NV;
  __shared__ float comx, comy, comz;

  if(threadIdx.x == 0){
    comx = 0.0f;
    comy = 0.0f;
    comz = 0.0f;
  }

  __syncthreads();

  if(fi < NV){
    atomicAdd(&comx,Verts[ci * NV + fi].X);
    atomicAdd(&comy,Verts[ci * NV + fi].Y);
    atomicAdd(&comz,Verts[ci * NV + fi].Z);
  }

  __syncthreads();

  if(threadIdx.x == 0){
    Cells[ci].COMX = comx/NV;
    Cells[ci].COMY = comy/NV;
    Cells[ci].COMZ = comz/NV;
  }
}

__device__ void cuVolumeForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset){
  //volume Forces
  int ci = blockIdx.x + offset;
  if(Cells[ci].Kv < 1e-6 || ci >= NCELLS){
    return;
  }
  if(threadIdx.x >= Cells[ci].ntriangles){
    return;
  }
  
  int NV = Cells[ci].NV;
  glm::ivec3 face = Faces[threadIdx.x];

  int i0 = ci * NV + face.x;
  int i1 = ci * NV + face.y;
  int i2 = ci * NV + face.z;

  glm::vec3 P0 = {Verts[i0].X, Verts[i0].Y, Verts[i0].Z};
  glm::vec3 P1 = {Verts[i1].X, Verts[i1].Y, Verts[i1].Z};
  glm::vec3 P2 = {Verts[i2].X, Verts[i2].Y, Verts[i2].Z};

  float Kv = Cells[ci].Kv;
  float V0 = Cells[ci].v0;
  float V = Cells[ci].Volume;
  float volumeStrain = (V / V0) - 1.0f;

  glm::vec3 COM = glm::vec3(Cells[ci].COMX, Cells[ci].COMY, Cells[ci].COMZ);
  glm::vec3 A = P1 - COM;
  glm::vec3 B = P2 - COM;
  glm::vec3 C = P0 - COM;

  // Gradient of volume wrt vertices
  glm::vec3 grad0 = glm::cross(A, B); // dV/dP0
  glm::vec3 grad1 = glm::cross(B, C); // dV/dP1
  glm::vec3 grad2 = glm::cross(C, A); // dV/dP2

  // Force is -Kv * strain * dV
  glm::vec3 f0 = -Kv * volumeStrain * grad0 / 6.0f;
  glm::vec3 f1 = -Kv * volumeStrain * grad1 / 6.0f;
  glm::vec3 f2 = -Kv * volumeStrain * grad2 / 6.0f;
  
  /*
  Verts[i0].Fx += f0.x;
  Verts[i0].Fy += f0.y;
  Verts[i0].Fz += f0.z;

  Verts[i1].Fx += f1.x;
  Verts[i1].Fy += f1.y;
  Verts[i1].Fz += f1.z;

  Verts[i2].Fx += f2.x;
  Verts[i2].Fy += f2.y;
  Verts[i2].Fz += f2.z;
  */


  atomicAdd(&Verts[i0].Fx,f0.x);
  atomicAdd(&Verts[i0].Fy,f0.y);
  atomicAdd(&Verts[i0].Fz,f0.z);

  atomicAdd(&Verts[i1].Fx,f1.x);
  atomicAdd(&Verts[i1].Fy,f1.y);
  atomicAdd(&Verts[i1].Fz,f1.z);

  atomicAdd(&Verts[i2].Fx,f2.x);
  atomicAdd(&Verts[i2].Fy,f2.y);
  atomicAdd(&Verts[i2].Fz,f2.z);
  
}

__device__ void cuSurfaceAreaForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset){
  //surface area forces
  int ci = blockIdx.x + offset;
  if(Cells[ci].Ka < 1e-6f || ci >= NCELLS){
    return;
  }
  glm::ivec3 face = Faces[threadIdx.x];

  int i0 = ci * Cells[ci].NV + face.x;
  int i1 = ci * Cells[ci].NV + face.y;
  int i2 = ci * Cells[ci].NV + face.z;

  auto inds = glm::ivec3(i0,i1,i2);

  glm::vec3 P0 = {Verts[i0].X, Verts[i0].Y, Verts[i0].Z};
  glm::vec3 P1 = {Verts[i1].X, Verts[i1].Y, Verts[i1].Z};
  glm::vec3 P2 = {Verts[i2].X, Verts[i2].Y, Verts[i2].Z};
  glm::vec3 lv[3] = {
    P1 - P0,
    P2 - P1,
    P0 - P2
  };
  for(int j=0; j < 3; j++){
    float len = glm::length(lv[j]);
    glm::vec3 u = lv[j]/len;
    float dli = len/Cells[ci].l0 - 1.0f;

    int j_prev = (j+2) % 3;
    glm::vec3 force = Cells[ci].Ka * (dli*u-(glm::length(lv[j_prev])/Cells[ci].l0 - 1.0f)*(lv[j_prev]/glm::length(lv[j_prev])));

    /*
    Verts[inds[j]].Fx += force.x;
    Verts[inds[j]].Fy += force.y;
    Verts[inds[j]].Fz += force.z;
    */
    atomicAdd(&Verts[inds[j]].Fx,force.x);
    atomicAdd(&Verts[inds[j]].Fy,force.y);
    atomicAdd(&Verts[inds[j]].Fz,force.z);
  }
}

__device__ void cuSurfaceAdhesionUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset){
  int ci = blockIdx.x + offset;
  if(Cells[ci].Ks < 1e-6f || ci > NCELLS){ 
    return;
  }
  glm::ivec3 face = Faces[threadIdx.x];
  int i0 = ci * Cells[ci].NV + face.x;
  int i1 = ci * Cells[ci].NV + face.y;
  int i2 = ci * Cells[ci].NV + face.z;
  glm::ivec3 inds = glm::ivec3(i0,i1,i2);

  glm::vec3 P0 = {Verts[i0].X, Verts[i0].Y, Verts[i0].Z};
  glm::vec3 P1 = {Verts[i1].X, Verts[i1].Y, Verts[i1].Z};
  glm::vec3 P2 = {Verts[i2].X, Verts[i2].Y, Verts[i2].Z};
  glm::mat3 Positions={P0,P1,P2};
  glm::vec3 flatten[3] = {P0,P1,P2};
  glm::mat3x3 forces{0.0f};

  auto A = P1 - P0;
  auto B = P2 - P0;
  auto COM = glm::vec3{Cells[ci].COMX,Cells[ci].COMY,Cells[ci].COMZ};

  for(int i =0; i<3;i++){
    flatten[i].z = 0.0f;
    if(Positions[i].z < flatten[i].z){
      forces[i].z += Cells[ci].Ks*pow((Positions[i].z - flatten[i].z),2);
    }
    float dist = glm::distance(flatten[i],Positions[i]);
    if((A.x*B.y - A.y*B.x < 0.0f) && dist < Cells[ci].r0/5.0f){
      float ftmp = (1.0 - dist/Cells[ci].l0);
      forces[i] += Cells[ci].Ks*ftmp*glm::normalize(flatten[i]-COM);
    }
  }
  for(int i=0;i<3;i++){
    /*
    Verts[inds[i]].Fx += forces[i].x;
    Verts[inds[i]].Fy += forces[i].y;
    Verts[inds[i]].Fz += forces[i].z;
    */
    atomicAdd(&Verts[inds[i]].Fx,forces[i].x);
    atomicAdd(&Verts[inds[i]].Fy,forces[i].y);
    atomicAdd(&Verts[inds[i]].Fz,forces[i].z);
  }
}

__global__ void cuSimpleSpringAttraction(int NCELLS, bool PBC, float L, float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, int offset){
  int ci = blockIdx.x+offset;
  if(ci > NCELLS) return;
  int vi = threadIdx.x + ci * Cells[ci].NV;
  float dist;
  float l0 = Cells[ci].l0;
  glm::vec3 PI = glm::vec3(Verts[vi].X,Verts[vi].Y,Verts[vi].Z);
  for(int cj = 0; cj < NCELLS;cj++){
    if(cj==ci){
      return;
    }
    for(int j=0;j<Cells[cj].NV;j++){
      int vj = threadIdx.x + cj * Cells[ci].NV;
      glm::vec3 PJ = glm::vec3(Verts[vj].X,Verts[vj].Y,Verts[vj].Z);
      glm::vec3 rij = PJ-PI;
      if(PBC){
        rij -= L * glm::round(rij/L);
      }
      dist = glm::sqrt(glm::dot(rij,rij));
      if(dist > l0*2){
        continue;
      }
      glm::vec3 force = Kat * 0.5f * ((dist/l0)-1.0f) * glm::normalize(rij);
      /*
      Verts[vi].Fx += force.x;
      Verts[vi].Fy += force.y;
      Verts[vi].Fz += force.z;
      Verts[vj].Fx -= force.x;
      Verts[vj].Fy -= force.y;
      Verts[vj].Fz -= force.z;
      */
      
      atomicAdd(&Verts[vi].Fx,force.x);
      atomicAdd(&Verts[vi].Fy,force.y);
      atomicAdd(&Verts[vi].Fz,force.z);
      atomicAdd(&Verts[vj].Fx,-force.x);
      atomicAdd(&Verts[vj].Fy,-force.y);
      atomicAdd(&Verts[vj].Fz,-force.z);
      
    } 
  }
}
__global__ void cuSlipBondAttraction(int NCELLS, bool PBC,float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, int offset){
  int ci = blockIdx.x + offset;
  if(ci > NCELLS) return;
  int vi = threadIdx.x + ci * Cells[ci].NV;
  float dist;
  float l0 = Cells[ci].l0;
  glm::vec3 PI = glm::vec3(Verts[vi].X,Verts[vi].Y,Verts[vi].Z);
  for(int cj = 0; cj < NCELLS;cj++){
    if(cj==ci){
      return;
    }
    for(int j=0;j<Cells[cj].NV;j++){
      int vj = threadIdx.x + cj * Cells[ci].NV;
      glm::vec3 PJ = glm::vec3(Verts[vj].X,Verts[vj].Y,Verts[vj].Z);
      glm::vec3 rij = PJ-PI;
      if(PBC){
        rij -= L * glm::round(rij/L);
      }
      dist = glm::sqrt(glm::dot(rij,rij));
      if(dist > l0*2){
        continue;
      }
      float ftmp = dist/Cells[ci].l0 * Kat;
      float f0 = Cells[ci].idealForce;
      float lifetime = glm::exp(-glm::abs(ftmp)/f0) + glm::exp(-glm::pow((glm::abs(ftmp)-f0)/f0,2));
      if(lifetime < 1e-8){
        continue;
      }
      ftmp /= lifetime;
      glm::vec3 force = 0.5f * ftmp * glm::normalize(rij);
      ///*
      atomicAdd(&Verts[vi].Fx , force.x);
      atomicAdd(&Verts[vi].Fy , force.y);
      atomicAdd(&Verts[vi].Fz , force.z);
      atomicAdd(&Verts[vj].Fx , -force.x);
      atomicAdd(&Verts[vj].Fy , -force.y);
      atomicAdd(&Verts[vj].Fz , -force.z);
      //*/
      /*
      Verts[vi].Fx += force.x;
      Verts[vi].Fy += force.y;
      Verts[vi].Fz += force.z;
      Verts[vj].Fx -= force.x;
      Verts[vj].Fy -= force.y;
      Verts[vj].Fz -= force.z;
      */
    } 
  }
}
  
__global__ void cuCatchBondAttraction(int NCELLS,bool PBC, float L ,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, int offset){
  int ci = blockIdx.x+offset;
  if(ci > NCELLS){
    return;
  }
  int vi = threadIdx.x + ci * Cells[ci].NV;
  float dist;
  float l0 = Cells[ci].l0;
  glm::vec3 PI = glm::vec3(Verts[vi].X,Verts[vi].Y,Verts[vi].Z);
  for(int cj = 0; cj < NCELLS;cj++){
    if(cj==ci){
      return;
    }
    for(int j=0;j<Cells[cj].NV;j++){
      int vj = threadIdx.x + cj * Cells[ci].NV;
      glm::vec3 PJ = glm::vec3(Verts[vj].X,Verts[vj].Y,Verts[vj].Z);
      glm::vec3 rij = PJ-PI;
      if(PBC){
        rij -= L * glm::round(rij/L);
      }
      dist = glm::sqrt(glm::dot(rij,rij));
      if(dist > l0*2){
        continue;
      }
      float ftmp = dist/Cells[ci].l0 * Kat;
      float f0 = Cells[ci].idealForce;
      float lifetime = glm::exp(-glm::abs(ftmp)/f0);
      if(lifetime < 1e-8) continue;
      ftmp /= lifetime;
      glm::vec3 force = 0.5f * ftmp * glm::normalize(rij);

      /*
      Verts[vi].Fx+=force.x;
      Verts[vi].Fy+=force.y;
      Verts[vi].Fz+=force.z;
      Verts[vj].Fx-=force.x;
      Verts[vj].Fy-=force.y;
      Verts[vj].Fz-=force.z;
      */

//      /*
      atomicAdd(&Verts[vi].Fx,force.x);
      atomicAdd(&Verts[vi].Fy,force.y);
      atomicAdd(&Verts[vi].Fz,force.z);
      atomicAdd(&Verts[vj].Fx,-force.x);
      atomicAdd(&Verts[vj].Fy,-force.y);
      atomicAdd(&Verts[vj].Fz,-force.z);
  //    */
    } 
  }
}
  

__global__ void cuResetForces3D(int NCELLS, int NV, cudaDPM::Vertex3D* Verts, int offset){
  int ci = blockIdx.x + offset;
  if(ci >= NCELLS) return;
  int vi = threadIdx.x + ci * NV;
  Verts[vi].Fx = 0.0f;
  Verts[vi].Fy = 0.0f;
  Verts[vi].Fz = 0.0f;
}

__global__ void cuEulerUpdate3D(float dt, int NCELLS, int NV, cudaDPM::Vertex3D *Verts, int offset){
    //Update Forces and Positions
  int ci = blockIdx.x + offset;
  if(ci >= NCELLS) return;
  int vi = threadIdx.x + ci * NV;
  ///*
  atomicAdd(&Verts[vi].X,dt*Verts[vi].Fx);
  atomicAdd(&Verts[vi].Y,dt*Verts[vi].Fy);
  atomicAdd(&Verts[vi].Z,dt*Verts[vi].Fz);
  //*/
  /*
  Verts[vi].X += Verts[vi].Fx*dt;
  Verts[vi].Y += Verts[vi].Fy*dt;
  Verts[vi].Z += Verts[vi].Fz*dt;
  */
  
}

__global__ void cuShapeForce3D(int NCELLS,cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset){
  int ci = blockIdx.x + offset;
  if(ci >= NCELLS) return;
  cuUpdateCellVolumes(NCELLS, Cells, Verts, Faces,offset);
  cuUpdateCOMS(NCELLS,Cells,Verts,offset);
  cuVolumeForceUpdate(NCELLS, Cells, Verts,Faces, offset);
  cuSurfaceAreaForceUpdate(NCELLS, Cells, Verts,Faces,offset);
  cuSurfaceAdhesionUpdate(NCELLS, Cells, Verts,Faces,offset);
}


//Need to add PBC handling
__global__ void cuRepellingForce3D(int NCELLS, bool PBC, float L, float Kc, cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces, int offset){
  int ci = blockIdx.x + offset;
  int NV = Cells[ci].NV;
  int vi = threadIdx.x + ci * NV;

  if(Kc < 1e-4 || ci >= NCELLS) return;

  glm::vec3 shift(0.0);
  glm::vec3 COM = glm::vec3(Cells[ci].COMX,Cells[ci].COMY,Cells[ci].COMZ);

  auto Force = glm::vec3{0.0f};
  glm::vec3 p = glm::vec3(Verts[vi].X, Verts[vi].Y, Verts[vi].Z);
  for(int cj=0;cj<NCELLS;cj++){
    auto COMJ = glm::vec3(Cells[cj].COMX,Cells[cj].COMY,Cells[cj].COMY);
    shift *=0;
    if(PBC) shift = L * glm::round((COM-COMJ)/L);
    auto newP = p - shift;
    if(ci==cj) continue;
    float winding_number = 0.0f;
    for(int fj=0;fj < Cells[cj].ntriangles; fj++){
      glm::ivec3 face = Faces[fj];
      glm::vec3 A = glm::vec3{Verts[face[0] + cj * NV].X,Verts[face[0]+cj*NV].Y,Verts[face[0]+cj*NV].Z};
      glm::vec3 B = glm::vec3{Verts[face[1] + cj * NV].X,Verts[face[1]+cj*NV].Y,Verts[face[1]+cj*NV].Z};
      glm::vec3 C = glm::vec3{Verts[face[2] + cj * NV].X,Verts[face[2]+cj*NV].Y,Verts[face[2]+cj*NV].Z};

      glm::vec3 a = A-newP;
      glm::vec3 b = B-newP;
      glm::vec3 c = C-newP;

      float la = length(a), lb = length(b) , lc = length(c);
      float denom = la*lb*lc + glm::dot(a,b)*lc + glm::dot(a,c)*lb + glm::dot(b,c)*la;
      float num = glm::dot(a,glm::cross(b,c));

      float omega = 2.0f * atan2(num,denom);
      winding_number += omega;
    }
    winding_number = fabs(winding_number);
    Force += winding_number * 0.5f * Kc * glm::normalize(COM-p);
  }

  /*Verts[vi].Fx += Force.x;
  Verts[vi].Fy += Force.y;
  Verts[vi].Fz += Force.z;
  */
  atomicAdd(&Verts[vi].Fx,Force.x);
  atomicAdd(&Verts[vi].Fy,Force.y);
  atomicAdd(&Verts[vi].Fz,Force.z);
}
