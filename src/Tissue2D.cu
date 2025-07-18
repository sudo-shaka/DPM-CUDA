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
#include"Cell2D.hpp"
#include"Tissue.hpp"
#include<vector>
#include<iostream>
#include<cmath>
#include<glm/glm.hpp>
#include<glm/vec3.hpp>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include"cudaKernel.cuh"

namespace cudaDPM{
  Tissue2D::Tissue2D(std::vector<cudaDPM::Cell2D> _Cells, float _phi0){
    phi0 = _phi0;
    Cells = _Cells;
    NCELLS = (int)Cells.size();
    VertDOF = 0;
    MaxNV = 0;
    float sumareas = 0.0;
    for(int ci=0;ci<NCELLS;ci++){
      VertDOF += Cells[ci].NV;
      sumareas += Cells[ci].GetArea();
      if(Cells[ci].NV > MaxNV){
        MaxNV = Cells[ci].NV;
      }
    }
    L = sqrt(sumareas)/_phi0;
    Kre = 1.0;
    U = 0.0;
  }

  void Tissue2D::EulerUpdate(int nsteps, float dt){
    int ci;
    cudaDPM::Vertex2D* VertsCUDA; //pointer of pointers for each cell verticies
    cudaDPM::Cell2D* CellCUDA;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    if(NCELLS > prop.maxBlocksPerMultiProcessor){
      std::cout << "Exceded number of cells allowed on cuda device" << std::endl;
      exit(0);
    }
    if(MaxNV> prop.maxThreadsPerBlock){
      std::cout << "Exceded number of verts allowed on cuda device" << std::endl;
      exit(0);
    }

    //Allocate mempory for the data on the CUDA device
    cudaError_t m1 = cudaMalloc((void **)&VertsCUDA, NCELLS * MaxNV * sizeof(cudaDPM::Vertex2D));
    cudaError_t m2 = cudaMalloc((void **)&CellCUDA,  NCELLS * sizeof(cudaDPM::Cell2D));
    if(m1 != cudaSuccess || m2 != cudaSuccess){
      std::cerr << cudaGetErrorString(m1) << " : " << cudaGetErrorString(m2) << std::endl;
    }

    //For each of the cells copy the vertex data to the memory we stored on the CUDA device
    cudaError_t mem;
    for(ci=0;ci<NCELLS;ci++){
      mem = cudaMemcpy((VertsCUDA+(ci*MaxNV)),Cells[ci].Verticies.data(),MaxNV * sizeof(cudaDPM::Vertex2D),cudaMemcpyHostToDevice);
      if(mem != cudaSuccess){
        std::cerr << cudaGetErrorString(mem) << std::endl;
        std::cerr << "[!] Error: cannot allocate vertex data to device : ";
      }
    }
    //Copy the cell data to the CUDA device
    mem = cudaMemcpy(CellCUDA,Cells.data(),NCELLS * sizeof(cudaDPM::Cell2D),cudaMemcpyHostToDevice);
    if(mem != cudaSuccess){
      std::cerr << cudaGetErrorString(mem) << std::endl;
      std::cerr << "[!] Error: cannot allocate cell data to cudaDevice : ";
    }

    //Start the Kernel
    cudaError_t cudaerr;
    for(int step=0;step<nsteps;step++){
      cuShapeForce2D<<<NCELLS,MaxNV>>>(dt,MaxNV,NCELLS,CellCUDA,VertsCUDA);
      cuRetractingForce2D<<<NCELLS,MaxNV>>>(dt,MaxNV,Kre,L,NCELLS,CellCUDA,VertsCUDA);
      cudaerr = cudaDeviceSynchronize();
      if(cudaerr != cudaSuccess){
        std::cerr << "[!] Error: cannot properly run cudaKernel : ";
        std::cerr << cudaGetErrorString(cudaerr) << std::endl;
      }
    }

    //Getting data back
    for(ci=0;ci<NCELLS;ci++){
      mem = cudaMemcpy(Cells[ci].Verticies.data(),(VertsCUDA+(ci*MaxNV)), Cells[ci].NV * sizeof(cudaDPM::Vertex2D),cudaMemcpyDeviceToHost);
    }
    if(mem != cudaSuccess){
      std::cerr << "[!] Error: cannot get data from cuda device : ";
      std::cerr << cudaGetErrorString(mem) << std::endl;
    }

    //Freeing up data
    cudaFree(CellCUDA);
    cudaFree(VertsCUDA);
  }

  void Tissue2D::disperse(){
    std::vector<float> X,Y,Fx,Fy;
    X.resize(NCELLS);Y.resize(NCELLS);
    Fx.resize(NCELLS); Fy.resize(NCELLS);
    float ri,xi,yi,xj,yj,dx,dy,rj,dist;
    float ux,uy,ftmp,fx,fy;
    int i,j, count=0;
    for(i=0;i<NCELLS;i++){
        X[i] = drand48() * L;
        Y[i] = drand48() * L;
    }
    float oldU = 100, dU = 100;
    while(dU > 1e-6){
      U = 0;
      for(i=0;i<NCELLS;i++){
          Fx[i] = 0.0;
          Fy[i] = 0.0;
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
              if(dist < 0.0) dist *= -1;
              if(dist <= (ri+rj)){
                ux = dx/dist;
                uy = dy/dist;
                ftmp = (1.0-dist/(ri+rj))/(ri+rj);
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
          X[i] += 0.01*Fx[i];
          Y[i] += 0.01*Fy[i];
        }
        dU = U-oldU;
        if(dU < 0.0)
            dU *= -1;
        oldU = U;
        count++;
        if(count > 1e4){
            break;
            std::cout << "Warning: dispersion may not have completed \n";
        }
    }
    for(int i=0;i<NCELLS;i++){
      for(j=0;j<Cells[i].NV;j++){
        Cells[i].Verticies[j].X = Cells[i].r0*(cos(2.0*M_PI*(j+1)/Cells[i].NV)) + X[i];
        Cells[i].Verticies[j].Y = Cells[i].r0*(sin(2.0*M_PI*(j+1)/Cells[i].NV)) + Y[i];
      }
    }
  }
}
