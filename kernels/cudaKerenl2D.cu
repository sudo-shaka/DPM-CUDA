#define GLM_ENAME_EXPERIMENTAL
#include "Cell2D.hpp"
#include "cudaKernel.cuh"
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
      dx -= L * round(dx / L);
      dy -= L * round(dy / L);
      rij = sqrt(dx*dx + dy*dy);
      rij = abs(sqrt(dx*dx + dy*dy));
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
