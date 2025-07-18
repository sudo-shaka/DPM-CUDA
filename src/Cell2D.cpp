/*
 * =====================================================================================
 *
 *       Filename:  Cell.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/01/2022 04:56:58 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Shaka,
 *   Organization:
 *
 * =====================================================================================
 */

#include<cmath>
#include"Cell2D.hpp"
#include<vector>
#include<iostream>

namespace cudaDPM{
  Vertex2D::Vertex2D(){
    X=0.0;
    Y=0.0;
    Fx = 0.0;
    Fy = 0.0;
    Vx = 0.0;
    Vy = 0.0;
  }
  Vertex2D::Vertex2D(float x, float y){
    X=x;
    Y=y;
    Fx = 0.0;
    Fy = 0.0;
    Vx = 0.0;
    Vy = 0.0;
    X=0.0;
    Y=0.0;
    Fx=0.0;
    Fy=0.0;
    Vx=0.0;
    Vy=0.0;
  }

  Cell2D::Cell2D(float x0, float y0, float _CalA0, int _NV, float _r0, float _Ka, float _Kl, float _Kb){
    NV = _NV;
    calA0 = _CalA0*(NV*tan(M_PI/NV)/M_PI);

    r0 = _r0;
    a0 = M_PI*(r0*r0);
    l0 = 2.0*sqrt(M_PI*calA0*a0)/NV;

    Kl = _Kl;
    Ka = _Ka;
    Kb = _Kb;
    Ks = 0.0;
    psi = 2*M_PI*drand48();
    Dr = 0.0;
    Ds = 0.0;
    v0 = 0.0;
    vmin = 0.0;

    Verticies.resize(NV);
    im1.resize(NV); ip1.resize(NV);

    for(int i=0;i<NV;i++){
      Verticies[i].X = r0*(cos(2.0*M_PI*(i+1)/NV)) + x0;
      Verticies[i].Y = r0*(sin(2.0*M_PI*(i+1)/NV)) + y0;
      im1[i] = i-1;
      ip1[i] = i+1;
    }

    im1[0] = NV-1;
    ip1[NV-1] = 0;
    Area = GetArea();
  }

  Cell2D::Cell2D(std::vector<Vertex2D> Verts){
    NV = Verts.size();
    Verticies.resize(NV);
    im1.resize(NV); ip1.resize(NV);
    for(int vi=0; vi<NV;vi++){
      Verticies[vi] = Verts[vi];
      ip1[vi] = vi-1;
      im1[vi] = vi+1;
    }
    im1[0] = NV-1;
    ip1[NV-1] = 0;
    Area = GetArea();
    if(NV > 30){
      std::cout << "Warning: Cells with > 30 Verticies sometimes have errors in the cuda calculations\n";
    }
  }

  void Cell2D::SetCellVelocity(float v){
    v0 = v;
    vmin = 1e-2*v0;
  }

  void Cell2D::UpdateDirectorDiffusion(float dt){
    float r1,r2,grv;
    r1 = drand48();
    r2 = drand48();
    grv = sqrt(-2.0*log(r1))*cos(2.0*M_PI*r2);
    psi += sqrt(2.0*dt*Dr)*grv;
  }

  float Cell2D::GetArea(){
    float Area = 0.0;
    int j = NV-1;
    for(int i=0; i<NV;i++){
      Area += 0.5 * ((Verticies[j].X + Verticies[i].X) * (Verticies[j].Y - Verticies[i].Y));
      j=i;
    }
    if(Area < 0.0){
      Area *= -1;
    }
    return Area;
  }
}
