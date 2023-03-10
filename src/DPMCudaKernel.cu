#include "../include/Cell.hpp"
#include<cmath>
#include <glm/geometric.hpp>
#include<stdio.h>
#include<glm/glm.hpp>
#include<glm/vec3.hpp>
#include<glm/mat3x3.hpp>
#include <glm/gtx/norm.hpp>

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

__global__ void cuShapeForce3D(float dt,int NCELLS,cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D *Verts, glm::ivec3 *Triangles){
  int ci = blockIdx.x;
  int fi = threadIdx.x;
  int NV = Cells[0].NV;
  float l0 = Cells[0].l0;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(ci < NCELLS && fi < Cells[ci].ntriangles){
    glm::ivec3 FaceIndex = Triangles[index];
    glm::vec3 P0 = {Verts[FaceIndex.x+(ci*NV)].X,Verts[FaceIndex.x+(ci*NV)].Y,Verts[FaceIndex.x+(ci*NV)].Z};
    glm::vec3 P1 = {Verts[FaceIndex.y+(ci*NV)].X,Verts[FaceIndex.y+(ci*NV)].Y,Verts[FaceIndex.y+(ci*NV)].Z};
    glm::vec3 P2 = {Verts[FaceIndex.z+(ci*NV)].X,Verts[FaceIndex.z+(ci*NV)].Y,Verts[FaceIndex.z+(ci*NV)].Z};
    glm::vec3 COM = {Cells[ci].COMX,Cells[ci].COMY,Cells[ci].COMZ};
    glm::mat3 Forces = {0,0,0,0,0,0,0,0,0}, Positions = {P0,P1,P2};
    int j;

      //Volume Force Update
    float VolumeStrain = (Cells[ci].Volume/Cells[ci].v0) - 1.0;
    glm::vec3 A = P1-P0;
    glm::vec3 B = P2-P0;
    glm::vec3 Direction = glm::normalize(glm::cross(A,B));

    for(j=0;j<3;j++){
      Forces[j] -= Cells[ci].Kv * 0.5f * VolumeStrain * Direction;
    }

    //SurfaceArea Force Update

    //Keep equalateral
    glm::mat3 lv,ulv;
    glm::vec3 length,dli;
    lv[0] = Positions[1] - Positions[0];
    lv[1] = Positions[2] - Positions[1];
    lv[2] = Positions[0] - Positions[2];
    length[0] = sqrt(glm::dot((lv[0]),(lv[0])));
    length[1] = sqrt(glm::dot((lv[1]),(lv[1])));
    length[2] = sqrt(glm::dot((lv[2]),(lv[2])));
    ulv[0] = lv[0]/length[0];
    ulv[1] = lv[1]/length[1];
    ulv[2] = lv[2]/length[2];
    dli[0] = length[0]/l0 - 1.0f;
    dli[1] = length[1]/l0 - 1.0f;
    dli[2] = length[2]/l0 - 1.0f;
    Forces[0] += Cells[ci].Ka * (dli[0]*ulv[0]-dli[2]*ulv[2]);
    Forces[1] += Cells[ci].Ka * (dli[1]*ulv[1]-dli[0]*ulv[0]);
    Forces[2] += Cells[ci].Ka * (dli[2]*ulv[2]-dli[1]*ulv[1]);

    //Just keep area constant
    /*float Area = 0.5 * glm::l2Norm(glm::cross((P1-P0),(P2-P0)));
    float AreaStrain=(Area/Cells[ci].a0) - 1.0;
    glm::vec3 center = (P0+P1+P2)/3.0f;
    Forces[0] += Cells[ci].Ka * 0.5f * AreaStrain * glm::normalize(center-P0);
    Forces[1] += Cells[ci].Ka * 0.5f * AreaStrain * glm::normalize(center-P1);
    Forces[2] += Cells[ci].Ka * 0.5f * AreaStrain * glm::normalize(center-P2);*/

    //Sticking to surface

    length = {Positions[0].z,Positions[1].z,Positions[2].z};
    glm::mat3 point = Positions;
    bool isnormaltosurface = ((A.x*B.y - A.y*B.x) < 0.0f);
    point[0].z = 0.0f;
    point[1].z = 0.0f;
    point[2].z = 0.0f;

    for(j=0;j<3;j++){
      if(length[j] < 0.0 && length[j] < l0){
        Forces[j].z -= Cells[ci].Ks * length[j]/l0;
      }
      else if(A.x*B.y - A.y*B.x < 0.0f && length[j] < l0){
        Forces[j] += Cells[ci].Ks * ((1.0f-length[j]/(l0))/l0) * glm::normalize(point[j]-COM);
      }
    }
    //Update Forces and Positions
    for(j=0;j<3;j++){
      Verts[FaceIndex[j]+(ci*NV)].Fx = Forces[j][0];
      Verts[FaceIndex[j]+(ci*NV)].Fy = Forces[j][1];
      Verts[FaceIndex[j]+(ci*NV)].Fz = Forces[j][2];
      //damped Euler update
      Verts[FaceIndex[j]+(ci*NV)].X += dt*Forces[j][0];
      Verts[FaceIndex[j]+(ci*NV)].Y += dt*Forces[j][1];
      Verts[FaceIndex[j]+(ci*NV)].Z += dt*Forces[j][2];
    }
  }

  __syncthreads();
}


__global__ void cuRepellingForce3D(float dt, int NCELLS, int NT, float L, float Kc,
                                   cudaDPM::Cell3D* Cells, cudaDPM::Vertex3D
                                   *Verts, glm::ivec3 *Triangles){

  int ci = blockIdx.x;
  int fi = threadIdx.x;
  int NV = Cells[0].NV;
  int cj,fj;
  float l0 = Cells[ci].l0;
  float dist;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(ci < NCELLS && fi < Cells[ci].ntriangles && Kc > 0.0001){
    glm::vec3 COM = {Cells[ci].COMX,Cells[ci].COMY,Cells[ci].COMZ};
    glm::ivec3 FaceIndexI = Triangles[index], FaceIndexJ;
    glm::mat3 Forces = {0,0,0,0,0,0,0,0,0}, PI, PJ;
    PI[0] = {Verts[FaceIndexI.x+(ci*NV)].X,Verts[FaceIndexI.x+(ci*NV)].Y,Verts[FaceIndexI.x+(ci*NV)].Z};
    PI[1] = {Verts[FaceIndexI.y+(ci*NV)].X,Verts[FaceIndexI.y+(ci*NV)].Y,Verts[FaceIndexI.y+(ci*NV)].Z};
    PI[2] = {Verts[FaceIndexI.z+(ci*NV)].X,Verts[FaceIndexI.z+(ci*NV)].Y,Verts[FaceIndexI.z+(ci*NV)].Z};
    glm::vec3 normali, FaceCenterI, FaceCenterJ, rij;

    FaceCenterI = (PI[0]+PI[1]+PI[2])/3.0f;

    normali = glm::cross((PI[1]-PI[0]),(PI[2]-PI[0]));
    for(cj=0;cj<NCELLS;cj++){
      if(ci != cj){
        for(fj=0;fj<NT;fj++){
          FaceIndexJ = Triangles[fj+(cj*NT)];
          PJ[0] = {Verts[FaceIndexJ.x+(cj*NV)].X,Verts[FaceIndexJ.x+(cj*NV)].Y,Verts[FaceIndexJ.x+(cj*NV)].Z};
          PJ[1] = {Verts[FaceIndexJ.y+(cj*NV)].X,Verts[FaceIndexJ.y+(cj*NV)].Y,Verts[FaceIndexJ.y+(cj*NV)].Z};
          PJ[2] = {Verts[FaceIndexJ.z+(cj*NV)].X,Verts[FaceIndexJ.z+(cj*NV)].Y,Verts[FaceIndexJ.z+(cj*NV)].Z};
          FaceCenterJ = (PJ[0]+PJ[1]+PJ[2])/3.0f;
          rij = FaceCenterJ-FaceCenterI;
          rij -= L*round(rij/L);
          dist = abs(sqrt(glm::dot(rij,rij)));
          if(glm::dot(normali,rij) < 0.0 && dist < l0){
            //Forces[0] += pow(dist,2.0f)*0.5f*Kc*glm::normalize(COM-PI[0]);
            //Forces[1] += pow(dist,2.0f)*0.5f*Kc*glm::normalize(COM-PI[1]);
            //Forces[2] += pow(dist,2.0f)*0.5f*Kc*glm::normalize(COM-PI[2]);
            Forces[0] += (dist/l0)*0.5f*Kc*glm::normalize(COM-PI[0]);
            Forces[1] += (dist/l0)*0.5f*Kc*glm::normalize(COM-PI[1]);
            Forces[2] += (dist/l0)*0.5f*Kc*glm::normalize(COM-PI[2]);
          }
        }
      }
    }
    //Update Forces and Positions
    for(int j=0;j<3;j++){
      Verts[FaceIndexI[j]+(ci*NV)].Fx = Forces[j][0];
      Verts[FaceIndexI[j]+(ci*NV)].Fy = Forces[j][1];
      Verts[FaceIndexI[j]+(ci*NV)].Fz = Forces[j][2];
      //damped Euler update
      Verts[FaceIndexI[j]+(ci*NV)].X += dt*Forces[j][0];
      Verts[FaceIndexI[j]+(ci*NV)].Y += dt*Forces[j][1];
      Verts[FaceIndexI[j]+(ci*NV)].Z += dt*Forces[j][2];
    }
  }

  __syncthreads();

}
