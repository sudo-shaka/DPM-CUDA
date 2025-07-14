#define GLM_ENAME_EXPERIMENTAL
#include "Cell.hpp"
#include "DPMCudaKernel.cuh"
#include<cmath>
#include <glm/geometric.hpp>
#include<glm/glm.hpp>
#include<glm/vec3.hpp>
#include<glm/mat3x3.hpp>

__global__ void cuShapeForce2D(float dt,int MaxNV, int NCELLS, cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D* Verts){
  int ci = blockIdx.x;
  int vi = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int indexm = index-1;
  int indexm2 = index-2;
  int indexp = index+1;
  int indexp2 = index+2;
  if(vi == Cells[ci].NV-1){
    indexp -= Cells[ci].NV;
    indexp2 = indexp+1;
  }
  else if(vi == Cells[ci].NV-2){
    indexp2 -= Cells[ci].NV;
  }
  if(vi == 0){
    indexm += (Cells[ci].NV);
    indexm2 = indexm - 1;
  }
  else if(vi == 1){
    indexm2 += (Cells[ci].NV);
  }

  float PartialArea = 0.0, areaStrain = 0.0;

  if(vi < Cells[ci].NV && ci < NCELLS){
    //ForceVars
    float Fxa = 0, Fya = 0, Fxb = 0, Fyb = 0, Fxp =0 ,Fyp=0;
    float Fys = 0, Fxs = 0;

    //PerimeterForceUpdate
    float lvxm,lvx;
    float lvym,lvy;
    float ulvxm,ulvx;
    float ulvym,ulvy;
    float dlim1, dli;
    float length, lengthm;
    float l0 = Cells[ci].l0;
    lvx = Verts[indexp].X - Verts[index].X;
    lvy = Verts[indexp].Y - Verts[index].Y;
    lvxm = Verts[index].X - Verts[indexm].X;
    lvym = Verts[index].Y - Verts[indexm].Y;
    length = sqrt(lvx*lvx + lvy*lvy);
    lengthm = sqrt(lvxm*lvxm + lvym*lvym);
    ulvx = lvx/length;
    ulvy = lvy/length;
    ulvxm = lvxm/lengthm;
    ulvym = lvym/lengthm;
    dli = length/l0 - 1.0;
    dlim1 = lengthm/l0 - 1.0;
    Fxp = Cells[ci].Kl*((sqrt(Cells[ci].a0)/l0))*(dli*ulvx- dlim1*ulvxm);
    Fyp = Cells[ci].Kl*((sqrt(Cells[ci].a0)/l0))*(dli*ulvy- dlim1*ulvym);

    //BendingForceUpdate
    float rho0 = sqrt(Cells[ci].a0);
    float fb = Cells[ci].Kb*(rho0/(l0*l0));
    float six, sixp, sixm;
    float siy, siyp, siym;
    six = lvx - lvxm;
    siy = lvy - lvym;
    sixp = (Verts[indexp2].X - Verts[indexp].X) - lvx;
    siyp = (Verts[indexp2].Y - Verts[indexp].Y) - lvy;
    sixm = lvxm - (Verts[indexm].X - Verts[indexm2].X);
    siym = lvym - (Verts[indexm].Y - Verts[indexm2].Y);
    Fxb = fb*(2.0*six - sixm - sixp);
    Fyb = fb*(2.0*siy - siym - siyp);

    //AreaForceUpdate
    Cells[ci].Area = 0.0;
    PartialArea = 0.5*((Verts[indexm].X + Verts[index].X)*(Verts[indexm].Y - Verts[index].Y));
    atomicAdd(&Cells[ci].Area, PartialArea);
    if(Cells[ci].Area < 0.0){Cells[ci].Area *= -1.0;}
    areaStrain = (Cells[ci].Area/Cells[ci].a0) - 1.0;
    Fxa = (Cells[ci].Ka/(sqrt(Cells[ci].a0)))*0.5*areaStrain*(Verts[indexm].Y-Verts[indexp].Y);
    Fya = (Cells[ci].Ka/(sqrt(Cells[ci].a0)))*0.5*areaStrain*(Verts[indexm].X-Verts[indexp].X);


    //Driving Force Update
    float Fxd=0.0, Fyd=0.0;
    if(Cells[ci].v0 != 0.0){
      float rx,ry,psiVi,v0tmp,rscale,dpsi;
      rx = Verts[index].X - Cells[ci].COMX;
      ry = Verts[index].Y - Cells[ci].COMY;
      psiVi = atan2(rx,ry);
      dpsi = psiVi - Cells[ci].psi;
      dpsi -= 2.0*M_PI*round(dpsi/(2.0*M_PI));
      v0tmp = Cells[ci].v0*exp(-(dpsi*dpsi)/(2.0*Cells[ci].Ds*Cells[ci].Ds)) + Cells[ci].vmin;
      rscale = sqrt(rx*rx + ry*ry);
      Fxd = v0tmp*(rx/rscale);
      Fyd = v0tmp*(ry/rscale);
    }

    //Update forces and Positions
    Verts[index].Fx = Fxa+Fxp+Fxb+Fxd+Fxs;
    Verts[index].Fy = Fya+Fyp+Fyb+Fyd+Fys;
    Verts[index].Vx = 0.5*dt*Verts[index].Fx;
    Verts[index].Vy = 0.5*dt*Verts[index].Fy;
    Verts[index].X += dt*Verts[index].Fx;
    Verts[index].Y += dt*Verts[index].Fy;

    __syncthreads();

  }
}


__global__ void cuRetractingForce2D(float dt,int MaxNV, float Kc, float L, int NCELLS, cudaDPM::Cell2D *Cells, cudaDPM::Vertex2D *Verts){

  int ci = blockIdx.x;
  int vi = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int indexm = index-1;
  int indexm2 = index-2;
  int indexp = index+1;
  int indexp2 = index+2;
  if(vi == Cells[ci].NV-1){
    indexp -= Cells[ci].NV;
    indexp2 = indexp+1;
  }
  else if(vi == Cells[ci].NV-2){
    indexp2 -= Cells[ci].NV;
  }
  if(vi == 0){
    indexm += (Cells[ci].NV);
    indexm2 = indexm - 1;
  }
  else if(vi == 1){
    indexm2 += (Cells[ci].NV);
  }
  float rij,xij,ftmp=0.0,dx,dy;

  if(vi < Cells[ci].NV && ci < NCELLS){
    //for all other cells, use crossing test to see if there is an overlap.
    int cj_vj_i;
    int cj_vj_j;
    bool overlaps = false;
    Cells[ci].COMX = 0.0;
    Cells[ci].COMY = 0.0;
    atomicAdd(&Cells[ci].COMX, Verts[index].X);
    atomicAdd(&Cells[ci].COMY, Verts[index].Y);
    Cells[ci].COMX /= Cells[ci].NV;
    Cells[ci].COMY /= Cells[ci].NV;
    int i,j;
    float dxi, dyi,dxj,dyj;

    __syncthreads();
    for(int cj=0;cj<NCELLS;cj++){
      overlaps = false;
      for(i=0,j = Cells[cj].NV-1; i<Cells[cj].NV; j = i++){
        cj_vj_i = (cj*MaxNV)+i;
        cj_vj_j = (cj*MaxNV)+j;
        dxi = Verts[index].X-Verts[cj_vj_i].X;
        dxj = Verts[index].X-Verts[cj_vj_j].X;
        dyi = Verts[index].Y-Verts[cj_vj_i].Y;
        dyj = Verts[index].Y-Verts[cj_vj_j].Y;
        if(abs(dxi) > L || abs(dxj) > L){
          dxi -= L*floor(dxi/L);
          dxj -= L*floor(dxj/L);
        }
        if(abs(dyi) > L || abs(dyj) > L){
          dyi -= L*round(dyi/L);
          dyj -= L*round(dyj/L);
        }

        if(ci != cj){
          if( ((dyi>0) != (dyj>0)) &&
              (0 < (dxj-dxi) * (0-dyi) / (dyj-dyi) + dxi) ){
            overlaps = !overlaps;
          }
        }
      }
      if(overlaps){
        break;
      }
    }

    if(overlaps){
      dx = Cells[ci].COMX - Verts[index].X;
      dy = Cells[ci].COMY - Verts[index].Y;
      rij = abs(sqrt(dx*dx + dy*dy));
      rij -= L*round(rij/L);
      xij = rij/(2*Cells[ci].r0);
      ftmp = Kc*(1-xij);
      Cells[ci].U += 0.5 * Kc * pow(1-xij,2);
      Verts[index].Fx += ftmp * (dx/rij);
      Verts[index].Fy += ftmp * (dy/rij);
      Verts[index].Vx = 0.5*dt*Verts[index].Fx;
      Verts[index].Vy = 0.5*dt*Verts[index].Fy;
      Verts[index].X += dt*(ftmp * (dx/rij));
      Verts[index].Y += dt*(ftmp * (dy/rij));
    }

    __syncthreads();
  }
}

__device__ void cuUpdateCellVolumes(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){
  int ci = blockIdx.x;
  int fi = threadIdx.x;
  if(ci >= NCELLS){
    return;
  }
  int NV = Cells[ci].NV;
  int ntri = Cells[ci].ntriangles;
  if(fi >= ntri){
    return;
  }

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

__device__ void cuUpdateCOMS(int NCELLS, cudaDPM::Cell3D *Cells,cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){
  int ci = blockIdx.x;
  int fi = threadIdx.x;
  if(ci >= NCELLS){
    return;
  }
  int NV = Cells[ci].NV;
  int ntri = Cells[ci].ntriangles;
  if(fi >= ntri){
    return;
  }
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

__device__ void cuVolumeForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){

  //volume Forces
  int ci = blockIdx.x;
  if(ci > NCELLS){
    return;
  }
  if(Cells[ci].Kv < 1e-6){
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

  Verts[i0].Fx += f0.x;
  Verts[i0].Fy += f0.y;
  Verts[i0].Fz += f0.z;

  Verts[i1].Fx += f1.x;
  Verts[i1].Fy += f1.y;
  Verts[i1].Fz += f1.z;

  Verts[i2].Fx += f2.x;
  Verts[i2].Fy += f2.y;
  Verts[i2].Fz += f2.z;
}

__device__ void cuSurfaceAreaForceUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){
  //surface area forces
  int ci = blockIdx.x;
  if(Cells[ci].Ka < 1e-6f){
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
    Verts[inds[j]].Fx += force.x;
    Verts[inds[j]].Fy += force.y;
    Verts[inds[j]].Fz += force.z;
  }
}

__device__ void cuSurfaceAdhesionUpdate(int NCELLS, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){
  int ci = blockIdx.x;
  if(Cells[ci].Ks < 1e-6f){ 
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
    Verts[inds[i]].Fx += forces[i].x;
    Verts[inds[i]].Fy += forces[i].y;
    Verts[inds[i]].Fz += forces[i].z;
  }
}

__global__ void cuSimpleSpringAttraction(int NCELLS, bool PBC, float L, float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts){
  int ci = blockIdx.x;
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
      Verts[vi].Fx -= force.x;
      Verts[vi].Fy -= force.y;
      Verts[vi].Fz -= force.z;
      Verts[vj].Fx += force.x;
      Verts[vj].Fy += force.y;
      Verts[vj].Fz += force.z;
    } 
  }
}
__global__ void cuSlipBondAttraction(int NCELLS, bool PBC,float L,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts){
  int ci = blockIdx.x;
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
      float lifetime = std::exp(-std::fabs(ftmp)/f0) + std::exp(-std::pow((std::fabs(ftmp)-f0)/f0,2));
      if(lifetime < 1e-8){
        continue;
      }
      ftmp /= lifetime;
      glm::vec3 force = 0.5f * ftmp * glm::normalize(rij);
      Verts[vi].Fx += force.x;
      Verts[vi].Fy += force.y;
      Verts[vi].Fz += force.z;
      Verts[vj].Fx -= force.x;
      Verts[vj].Fy -= force.y;
      Verts[vj].Fz -= force.z;
    } 
  }
}
  
__global__ void cuCatchBondAttraction(int NCELLS,bool PBC, float L ,float Kat, cudaDPM::Cell3D *Cells, cudaDPM::Vertex3D *Verts){
  int ci = blockIdx.x;
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
      float lifetime = std::exp(-std::fabs(ftmp)/f0);
      if(lifetime < 1e-8) continue;
      ftmp /= lifetime;
      glm::vec3 force = 0.5f * ftmp * glm::normalize(rij);
      Verts[vi].Fx += force.x;
      Verts[vi].Fy += force.y;
      Verts[vi].Fz += force.z;
      Verts[vj].Fx -= force.x;
      Verts[vj].Fy -= force.y;
      Verts[vj].Fz -= force.z;
    } 
  }
}
  

__global__ void cuResetForces3D(int NCELLS, int NV, cudaDPM::Vertex3D* Verts){
  int ci = blockIdx.x;
  if(ci >= NCELLS) return;
  int vi = threadIdx.x + ci * NV;
  Verts[vi].Fx = 0.0f;
  Verts[vi].Fy = 0.0f;
  Verts[vi].Fz = 0.0f;
  __syncthreads();
}

__global__ void cuEulerUpdate3D(float dt, int NCELLS, int NV, cudaDPM::Vertex3D *Verts){
    //Update Forces and Positions
  int ci = blockIdx.x;
  if(ci >= NCELLS) return;
  int vi = threadIdx.x + ci * NV;
  Verts[vi].X += dt*Verts[vi].Fx;
  Verts[vi].Y += dt*Verts[vi].Fy;
  Verts[vi].Z += dt*Verts[vi].Fz;
  __syncthreads();
  
}

__global__ void cuShapeForce3D(int NCELLS,cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){
  cuUpdateCellVolumes(NCELLS, Cells, Verts, Faces);
  cuUpdateCOMS(NCELLS,Cells,Verts,Faces);
  cuVolumeForceUpdate(NCELLS, Cells, Verts,Faces);
  cuSurfaceAreaForceUpdate(NCELLS, Cells, Verts,Faces);
  cuSurfaceAdhesionUpdate(NCELLS, Cells, Verts,Faces);
}


//Need to add PBC handling
__global__ void cuRepellingForce3D(int NCELLS, bool PBC, float L, float Kc, cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Faces){

  int ci = blockIdx.x;
  int vi = threadIdx.x + ci * Cells[ci].NV;

  if(ci >= NCELLS || Kc < 1e-4) return;

  glm::vec3 shift(0.0);
  glm::vec3 COM = glm::vec3(Cells[ci].COMX,Cells[ci].COMY,Cells[ci].COMZ);

  auto Force = glm::vec3{0.0f};
  glm::vec3 p = glm::vec3(Verts[vi].X, Verts[vi].Y, Verts[vi].Z);
  for(int cj=0;cj<NCELLS;cj++){
    auto COMJ = glm::vec3(Cells[cj].COMX,Cells[cj].COMY,Cells[cj].COMY);
    if(PBC) shift = L * glm::round((COM-COMJ)/L);
    auto newP = p - shift;
    if(ci==cj) continue;
    float winding_number = 0.0f;
    for(int fj=0;fj < Cells[cj].ntriangles; fj++){
      glm::ivec3 face = Faces[fj+cj*Cells[cj].ntriangles];
      glm::vec3 A = glm::vec3{Verts[face[0] + cj * Cells[cj].NV].X,Verts[face[0]+cj*Cells[cj].NV].Y,Verts[face[0]+cj*Cells[cj].NV].Z};
      glm::vec3 B = glm::vec3{Verts[face[1] + cj * Cells[cj].NV].X,Verts[face[1]+cj*Cells[cj].NV].Y,Verts[face[1]+cj*Cells[cj].NV].Z};
      glm::vec3 C = glm::vec3{Verts[face[2] + cj * Cells[cj].NV].X,Verts[face[2]+cj*Cells[cj].NV].Y,Verts[face[2]+cj*Cells[cj].NV].Z};

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

  Verts[vi].Fx += Force.x;
  Verts[vi].Fy += Force.y;
  Verts[vi].Fz += Force.z;
}
