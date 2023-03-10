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

#include"../include/Cell.hpp"
#include"../include/Tissue.hpp"
#include<vector>
#include<thread>
#include<iostream>
#include<cmath>
#include<glm/glm.hpp>
#include<glm/vec3.hpp>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include"../include/DPMCudaKernel.cuh"

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
    Kc = 1.0;
    U = 0.0;
  }

  void Tissue2D::EulerUpdate(int nsteps, float dt){
    int ci;
    cudaDPM::Vertex2D* VertsCUDA; //pointer of pointers for each cell verticies
    cudaDPM::Cell2D* CellCUDA;

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
      cuRetractingForce2D<<<NCELLS,MaxNV>>>(dt,MaxNV,Kc,L,NCELLS,CellCUDA,VertsCUDA);
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

  Tissue3D::Tissue3D(std::vector<cudaDPM::Cell3D> _Cells, float _phi0){
    Cells=_Cells;
    int nv = Cells[0].NV;
    for(cudaDPM::Cell3D c : Cells){
      if((int)c.NV != nv){
        std::cerr << "[!] Error, all cells must have the same number of verticies" << std::endl;
        exit(0);
      }
    }
    phi0 = _phi0;
    NCELLS = Cells.size();
    float volume = 0.0;
    VertDOF = Cells[0].NV * NCELLS;
    TriDOF = Cells[0].ntriangles * NCELLS;
    for(int i=0;i<NCELLS;i++){
      volume += Cells[i].GetVolume();
    }
    L=cbrt(volume)/phi0;
  }

  void Tissue3D::disperse2D(){
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
    float oldU = 100.0f,dU = 100.0f;
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

  void Tissue3D::EulerUpdate(int nsteps, float dt){
    int ci,NV = Cells[0].NV, NT = Cells[0].ntriangles;
    std::vector<std::thread> threads;
    if(NT > 1024){
      std::cerr << "[!] Error: Euler update cannot be completed. Greater than 1024 threads...\n";
      return;
    }

    //initializing pointers to reference host memorty for cuda
    cudaDPM::Cell3D* CellsCuda;
    cudaDPM::Vertex3D* VertsCuda;
    glm::ivec3* TriCuda;

    std::vector<cudaError_t> errors;
    errors.resize(3);

    //allocating memory for cuda
    errors[0] = cudaMalloc((void **)&CellsCuda, NCELLS  * sizeof(cudaDPM::Cell3D));
    errors[1] = cudaMalloc((void **)&VertsCuda, VertDOF * sizeof(cudaDPM::Vertex3D));
    errors[2] = cudaMalloc((void **)&TriCuda  , TriDOF  * sizeof(glm::ivec3));

    //Checking for errors
    for(auto& error : errors){
      if(error != cudaSuccess){
        std::cerr << cudaGetErrorString(error) << std::endl;
        exit(0);
      }
    }

    //Give data to cuda
    for(ci = 0; ci<NCELLS; ci++){
      cudaMemcpy(TriCuda+(NT*ci),Cells[ci].FaceIndices.data(),NT * sizeof(glm::ivec3),cudaMemcpyHostToDevice);
      cudaMemcpy(VertsCuda+(NV*ci),Cells[ci].Verticies.data(),NV * sizeof(cudaDPM::Vertex3D),cudaMemcpyHostToDevice);
    }

    for(int s=0; s<nsteps;s++){
      //Need to update volume and center for each timestep (calculated on CPU)
      std::vector<std::thread> threads;
      threads.resize(NCELLS);
      for(ci=0;ci<NCELLS;ci++){
        threads[ci] = std::thread(&cudaDPM::Cell3D::UpdateCOM,&this->Cells[ci]);
      }

      for(auto& th : threads){
        th.join();
      }

      for(ci=0;ci<NCELLS;ci++){
        threads[ci] = std::thread(&cudaDPM::Cell3D::UpdateVolume,&this->Cells[ci]);
      }

      for(auto& th : threads){
        th.join();
      }

      cudaMemcpy(CellsCuda,Cells.data(),NCELLS * sizeof(cudaDPM::Cell3D), cudaMemcpyHostToDevice);
      cuShapeForce3D<<<NCELLS,NT>>>(dt,NCELLS,CellsCuda,VertsCuda,TriCuda);
      cuRepellingForce3D<<<NCELLS,NT>>>(dt,NCELLS,NT,L,Kc,CellsCuda,VertsCuda,TriCuda);
      cudaMemcpy(Cells.data(),CellsCuda,NCELLS * sizeof(cudaDPM::Cell3D),cudaMemcpyDeviceToHost);
      for(ci = 0; ci<NCELLS; ci++)
        cudaMemcpy(Cells[ci].Verticies.data(),VertsCuda+(NV*ci),NV * sizeof(cudaDPM::Vertex3D),cudaMemcpyDeviceToHost);
    }

    //Free mem on cuda
    cudaFree(CellsCuda); cudaFree(VertsCuda); cudaFree(TriCuda);
  }
}
