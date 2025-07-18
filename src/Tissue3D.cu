/*
      }
 * =====================================================================================
 *
 *       Filename:  Tissue.cpp
 *
 *    Description:  cudaDPM Tissue interactions and integrators
 *
 *        Version:  1.0
 *        Created:  06/02/2022 09:03:23 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Shaka X,
 *   Organization:  Yale University
 *
 * =====================================================================================
 */

#define GLM_ENAME_EXPERIMENTAL
#include"Cell3D.hpp"
#include"Tissue.hpp"
#include<vector>
#include<array>
#include<iostream>
#include<fstream>
#include<cmath>
#include<glm/glm.hpp>
#include<glm/vec3.hpp>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include"cudaKernel.cuh"

namespace cudaDPM{
  Tissue3D::Tissue3D(std::vector<cudaDPM::Cell3D> _Cells, float _phi0){
    Cells=_Cells;
    PBC = false;
    int nv = Cells[0].NV;
    for(cudaDPM::Cell3D c : Cells){
      if((int)c.NV != nv){
        std::cerr << "[!] Error, all cells must have the same number of verticies" << std::endl;
        exit(0);
      }
    }
    setAttractionMethod("SimpleSpring");
    phi0 = _phi0;
    NCELLS = Cells.size();
    float volume = 0.0;
    VertDOF = Cells[0].NV * NCELLS;
    for(int i=0;i<NCELLS;i++){
      volume += Cells[i].GetVolume();
    }
    L=cbrt(volume)/phi0;
  }

  void Tissue3D::disperse2D(){
    PBC = true;
    float sumArea = 0.0f;
    for(int i=0;i<NCELLS;i++){
      sumArea += M_PI*Cells[i].r0*Cells[i].r0;
    }
    L = sqrt(sumArea)/phi0;
    std::vector<float> X,Y,Fx,Fy;
    X.resize(NCELLS);
    Y.resize(NCELLS);
    Fx.resize(NCELLS);
    Fy.resize(NCELLS);
    float ri,rj,yi,yj,xi,xj,dx,dy,dist;
    float ux,uy,ftmp,fx,fy;
    int i,j,count;
    for(i=0;i<NCELLS;i++){
      X[i] = drand48() * L;
      Y[i] = drand48() * L;
    }
    float oldU = MAXFLOAT,dU = MAXFLOAT;
    float U = 0.0;
    count = 0;
    while(dU > 1e-6){
      U=0.0f;
      for(i=0;i<NCELLS;i++){
        Fx[i] = 0.0f;
        Fy[i] = 0.0f;
      }
      for(i=0;i<NCELLS;i++){
        xi = X[i];
        yi = Y[i];
        ri = Cells[i].r0;
        for(j=0;j<NCELLS;j++){
            if(j != i){
              xj = X[j];
              yj = Y[j];
              rj = Cells[j].r0;
              dx = xj-xi;
              dx -= L*round(dx/L);
              dy = yj-yi;
              dy -= L*round(dy/L);
              dist = sqrt(dx*dx + dy*dy);
              if(dist < 0.0f)
                  dist *= -1;
              if(dist <= (ri+rj)){
                ux = dx/dist;
                uy = dy/dist;
                ftmp = (1.0f-dist/(ri+rj))/(ri+rj);
                fx = ftmp*ux;
                fy = ftmp*uy;
                Fx[i] -= fx;
                Fy[i] -= fy;
                Fy[j] += fy;
                Fx[j] += fx;
                U += 0.5*(1-(dist/(ri+rj))*(1-dist/(ri+rj)));
              }
            }
          }
        }
      for(int i=0; i<NCELLS;i++){
        X[i] += 0.01f*Fx[i];
        Y[i] += 0.01f*Fy[i];
      }
      dU = U-oldU;
      if(dU < 0.0f)
          dU *= -1.0f;
        oldU = U;
        count++;
        if(count > 1e5){
          std::cerr << "Warning: Max timesteps for dispersion reached"  << std::endl;
          break;
        }
    }
    for(i=0; i<NCELLS; i++){
      Cells[i].UpdateCOM();
      for(j=0;j<Cells[i].NV;j++){
        Cells[i].Verticies[j].X -= Cells[i].COMX;
        Cells[i].Verticies[j].Y -= Cells[i].COMY;
        Cells[i].Verticies[j].X += X[i];
        Cells[i].Verticies[j].Y += Y[i];
      }
    }
  }

  void Tissue3D::disperse3D(){
    PBC = true;
    std::vector<glm::vec3> centers;
    std::vector<glm::vec3> forces;
    glm::vec3 rij;
    centers.resize(NCELLS);
    forces.resize(NCELLS);
    int i,j,count=0;
    float ftmp;
    for(i=0;i<NCELLS;i++){
      centers[i].x = drand48() * L;
      centers[i].y = drand48() * L;
      centers[i].z = drand48() * L;
    }
    double oldU = 100, dU = 100, U, dist;
    while(dU > 1e-6){
      U = 0;
      for(i=0;i<NCELLS;i++){
        forces[i] = {0,0,0};
      }
      for(i=0;i<NCELLS;i++){
        for(j=0;j<NCELLS;j++){
          if(i!=j){
            rij = centers[j] - centers[i];
            rij -= L*round(rij/L);
            dist = sqrt(glm::dot(rij,rij));
            if(dist < 0.0){
              dist *= -1;
            }
            if(dist < (Cells[i].r0 + Cells[j].r0)){
              ftmp = (1-dist/(Cells[i].r0+Cells[j].r0)/(Cells[i].r0+Cells[j].r0));
              forces[i] -= ftmp*glm::normalize(rij);
              forces[j] += ftmp*glm::normalize(rij);
              U += 0.5*(1-(dist/(Cells[i].r0+Cells[j].r0))*(1-dist/(Cells[i].r0+Cells[j].r0)));
            }
          }
        }
      }
      for(i=0;i<NCELLS;i++){
        centers[i] += 0.01f*forces[i];
      }
      dU = U - oldU;
      if(dU < 0.0){
        dU *=-1;
      }
      oldU = U;
      count++;
      if(count > 1e5){
        std::cerr << "Warning: Max timesteps for dispersion exceeded" << std::endl;
        break;
      }
    }
    for(i=0;i<NCELLS;i++){
      Cells[i].UpdateCOM();
      for(j=0;j<Cells[i].NV;j++){
        Cells[i].Verticies[j].X -= Cells[i].COMX;
        Cells[i].Verticies[j].Y -= Cells[i].COMY;
        Cells[i].Verticies[j].Z -= Cells[i].COMZ;
        Cells[i].Verticies[j].X += centers[i].x;
        Cells[i].Verticies[j].Y += centers[i].y;
        Cells[i].Verticies[j].Z += centers[i].z;
      }
    }
  }

  void Tissue3D::setAttractionMethod(std::string method){
    attMethodName = method;
    auto it = methods.find(attMethodName);
    if(it != methods.end()){
      attractionMethod = [this, func = it->second](int offset){
        func(this,offset);
      };
    }
    else{
      std::cerr << "ERROR! Unknown method: " << attMethodName << ". Exiting.." << std::endl;
      exit(0);
    }
  }

  void Tissue3D::SimpleSpringAttraction(int offset){
    if(!CellsCuda || !VertsCuda){
      std::cout << "Cells or Verts cudaMemory is not allocated" << std::endl;
      return;
    }
    int NV = Cells[0].NV;
    cuSimpleSpringAttraction<<<NCELLS,NV>>>(NCELLS,PBC,L,Kat,CellsCuda,VertsCuda,offset);
  }

  void Tissue3D::CatchBondAttraction(int offset){
    if(!CellsCuda || !VertsCuda){
      std::cout << "Cells or Verts cudaMemory is not allocated" << std::endl;
      return;
    }
    int NV = Cells[0].NV;
    cuCatchBondAttraction<<<NCELLS,NV>>>(NCELLS,PBC,L,Kat,CellsCuda,VertsCuda,offset);
  }

  void Tissue3D::SlipBondAttraction(int offset){
    if(!CellsCuda || !VertsCuda || !VertsCuda){
      std::cout << "Cells or Verts cudaMemory is not allocated" << std::endl;
      return;
    }
    int NV = Cells[0].NV;
    cuCatchBondAttraction<<<NCELLS,NV>>>(NCELLS,PBC,L,Kat,CellsCuda,VertsCuda,offset);
  }

  void Tissue3D::SetUpCudaMemory(){
    for(int ci=0; ci<NCELLS;ci++){
      if(Cells[ci].NV != Cells[0].NV || Cells[ci].ntriangles != Cells[0].ntriangles){
      std::cerr << "[!]error: all cells must have the same number of verts and faces"<< std::endl; 
        exit(0);
      }
    }
    VertDOF = Cells[0].NV * NCELLS;

    std::array<cudaError_t,3> errors;
    errors[0] = cudaMalloc((void **)&CellsCuda, NCELLS  * sizeof(cudaDPM::Cell3D));
    errors[1] = cudaMalloc((void **)&VertsCuda, VertDOF * sizeof(cudaDPM::Vertex3D));
    errors[2] = cudaMalloc((void **)&TriCuda  , Cells[0].ntriangles  * sizeof(glm::ivec3));
    //Checking for errors
    for(auto& error : errors){
      if(error != cudaSuccess){
        std::cerr << cudaGetErrorString(error) << std::endl;
        cudaFree(VertsCuda); cudaFree(CellsCuda); cudaFree(TriCuda);
        exit(0);
      }
    }

    for(int ci = 0; ci<NCELLS; ci++){
      int NT = Cells[ci].ntriangles; int NV = Cells[ci].NV;
      cudaMemcpy(VertsCuda+(NV*ci),Cells[ci].Verticies.data(),NV * sizeof(cudaDPM::Vertex3D),cudaMemcpyHostToDevice);
    }
    cudaMemcpy(TriCuda,Cells[0].FaceIndices.data(),Cells[0].ntriangles * sizeof(glm::ivec3),cudaMemcpyHostToDevice);
    cudaMemcpy(CellsCuda,Cells.data(),NCELLS * sizeof(cudaDPM::Cell3D),cudaMemcpyHostToDevice);

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "\nCUDA Error: SETUP MEMCPY: " << cudaGetErrorString(err) << std::endl;
      cudaFree(VertsCuda); cudaFree(CellsCuda); cudaFree(TriCuda);
      exit(0);
    }
  }

  void Tissue3D::ExtractCudaMemory(){
    if(!CellsCuda || !TriCuda || !VertsCuda){
      std::cerr << "Cuda memory not allocated" << std::endl;
      return;
    }
    int NV = Cells[0].NV; int NT = Cells[0].ntriangles;
    for(int ci = 0; ci<NCELLS; ci++){
        cudaMemcpy(Cells[ci].Verticies.data(),VertsCuda+(NV*ci),NV * sizeof(cudaDPM::Vertex3D),cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(Cells.data(),CellsCuda,NCELLS * sizeof(cudaDPM::Cell3D),cudaMemcpyDeviceToHost);

  }

  void Tissue3D::EulerUpdate(int nsteps, float dt){
    SetUpCudaMemory();
    //Give data to cuda
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    if(Cells[0].ntriangles > prop.maxThreadsPerBlock){
      std::cout << "\nExceded number of faces allowed on cuda device" << std::endl;
      cudaFree(VertsCuda); cudaFree(CellsCuda); cudaFree(TriCuda);
      exit(0);
    }
    int  NV = Cells[0].NV; int NT=Cells[0].ntriangles;
    for(int s=0; s<nsteps;s++){
      for(int offset=0; offset < NCELLS*prop.maxThreadsPerBlock; offset+=prop.maxThreadsPerBlock){
        cuResetForces3D<<<NCELLS,NV>>>(NCELLS,NV, VertsCuda,offset);
        cudaDeviceSynchronize();
        cuShapeForce3D<<<NCELLS,NT>>>(NCELLS,CellsCuda,VertsCuda,TriCuda,offset);
        cuRepellingForce3D<<<NCELLS,NV>>>(NCELLS,PBC, L,Kre,CellsCuda,VertsCuda,TriCuda,offset);
        if(Kat > 0) attractionMethod(offset);
        cudaDeviceSynchronize();
        cuEulerUpdate3D<<<NCELLS,NV>>>(dt,NCELLS,NV,VertsCuda,offset);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << "\nCUDA Error: INTEGRATION: " << cudaGetErrorString(err) << std::endl;
        }
      }
    }
    ExtractCudaMemory();
    //Free mem on cuda
    cudaFree(CellsCuda); cudaFree(VertsCuda); cudaFree(TriCuda);
    CellsCuda = NULL; VertsCuda = NULL; TriCuda = NULL;
  }

  void Tissue3D::exportForcesToCSV(std::string filename){
    std::ofstream file;
    file.open(filename);
    for(int i=0;i<NCELLS;i++){
      file << "Cell_" << i << ",X,";
      for(auto& vert : Cells[i].Verticies){
        file << vert.Fx << ",";
      }
      file << "\nCell_" << i << ",Y,";
      for(auto& vert : Cells[i].Verticies){
        file << vert.Fy << ",";
      }
      file << "\nCell_" << i << ",Z,";
      for(auto& vert : Cells[i].Verticies){
        file << vert.Fz << ",";
      }
      file << std::endl;
    }
    file.close();

  }

  void Tissue3D::exportPositionsToCSV(std::string filename){
    std::ofstream file;
    file.open(filename);
    for(int i=0;i<NCELLS;i++){
      file << "Cell_" << i << ",X,";
      for(auto& vert : Cells[i].Verticies){
        file << vert.X << ",";
      }
      file << "\nCell_" << i << ",Y,";
      for(auto& vert : Cells[i].Verticies){
        file << vert.Y << ",";
      }
      file << "\nCell_" << i << ",Z,";
      for(auto& vert : Cells[i].Verticies){
        file << vert.Z << ",";
      }
      file << std::endl;
    }
    file.close();
  }
}
