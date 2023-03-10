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
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include<cmath>
#include"../include/Cell.hpp"
#include<vector>
#include<iostream>
#include"../include/GeometricFunctions.hpp"

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

  Vertex3D::Vertex3D(){
    X=0.0;
    Y=0.0;
    Z=0.0;
    Fx=0.0;
    Fy=0.0;
    Fz = 0.0;
    Vx=0.0;
    Vy=0.0;
    Vz=0.0;
  }
  Vertex3D::Vertex3D(float x, float y,float z){
    X=x;
    Y=y;
    Z=z;
    Fx=0.0;
    Fy=0.0;
    Fz = 0.0;
    Vx=0.0;
    Vy=0.0;
    Vz=0.0;
  }


  float Vertex3D::normPos(){
    return sqrt(X*X + Y*Y + Z*Z);
  }

  void Vertex3D::normalizePos(){
    float sum = X + Y + Z;
    X /= sum;
    Y /= sum;
    Z /= sum;
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
    if(NV > 30){
      std::cout << "Warning: Cells with > 30 Verticies sometimes have errors in the cuda calculations\n";
    }
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

  Cell3D::Cell3D(std::vector<float> X,
                 std::vector<float> Y,
                 std::vector<float> Z,
                 std::vector<std::vector<int>> Triangles,
                 float _v0, float _a0, float _Kv, float _Ka){
    NDIM = 3;
    int sizeX = X.size();
    int sizeY = X.size();
    int sizeZ = X.size();
    int i;

    if(sizeX != sizeY || sizeX != sizeZ || sizeZ != sizeY){
      std::cerr << "Lengths of X, Y, and Z must be equal" << std::endl;
      exit(1);
    }
    NV = sizeX;
    Verticies.resize(sizeX);
    for(i=0; i<sizeX; i++){
      Verticies[i].X = X[i];
      Verticies[i].Y = Y[i];
      Verticies[i].Z = Z[i];
    }
    ntriangles = Triangles.size();
    FaceIndices.resize(ntriangles);
    for(i=0;i<ntriangles;i++){
      if(Triangles[i].size() != 3){
        std::cerr << "Triangle index list must contain only 3 values each" << std::endl;
        exit(1);
      }
      FaceIndices[i].x = Triangles[i][0];
      FaceIndices[i].y = Triangles[i][1];
      FaceIndices[i].z = Triangles[i][2];
    }
    v0 = _v0;
    a0 = _a0;
    Kv = _Kv;
    Ka = _Ka;
    l0 = sqrt((4*a0)/sqrt(3));
   // Kb = _Kb;
  }

  Cell3D::Cell3D(std::vector<float> X, std::vector<float> Y, std::vector<float> Z,
                 float _v0, float _a0, float _Kv, float _Ka){
    NDIM = 3;
    int i, sizeX = X.size(), sizeY = Y.size(), sizeZ = Z.size();
    if(sizeX != sizeY || sizeX != sizeZ || sizeZ != sizeY){
      std::cerr << "Lengths of X, Y, and Z must be equal" << std::endl;
      exit(1);
    }
    NV = sizeX;
    Verticies.resize(sizeX);
    for(i=0; i<sizeX; i++){
      Verticies[i].X = X[i];
      Verticies[i].Y = Y[i];
      Verticies[i].Z = Z[i];
    }


    v0 = _v0;
    a0 = _a0;
    Kv = _Kv;
    Ka = _Ka;
    l0 = sqrt((4*a0)/sqrt(3));
  }

  Cell3D::Cell3D(float x0, float y0, float z0, float _CalA0, int f, float _r0, float _Kv, float _Ka){
    if(f > 3){
      std::cerr << "[!] Error: There is a maximum or 3 Icoshere recursions allowed" << std::endl;
      exit(0);
    }
    NDIM = 3;
    calA0 = _CalA0;
    r0 = _r0;
    Kv = _Kv;
    Ka = _Ka;
    //Kb = _Kb;
    Ks = 0.0;
    int i,j,steps,a,b,c;
    float t = (1+sqrt(5)/2);
    NV = 12;
    Verticies.resize(NV);
    SetVertexPos(0, -1, t, 0);
    SetVertexPos(1,  1, t, 0);
    SetVertexPos(2, -1,-t, 0);
    SetVertexPos(3,  1,-t, 0);

    SetVertexPos(4, 0, -1,  t);
    SetVertexPos(5, 0,  1,  t);
    SetVertexPos(6, 0, -1, -t);
    SetVertexPos(7, 0,  1, -t);

    SetVertexPos(8, t,  0, -1);
    SetVertexPos(9, t,  0,  1);
    SetVertexPos(10,-t, 0, -1);
    SetVertexPos(11,-t, 0,  1);
    float norm;
    for(i=0;i<12;i++){
      norm = Verticies[i].normPos();
      Verticies[i].X /= norm;
      Verticies[i].Y /= norm;
      Verticies[i].Z /= norm;
    }

    AddFaceIndex(0, 11, 5);
    AddFaceIndex(0, 5, 1);
    AddFaceIndex(0, 1, 7);
    AddFaceIndex(0, 7, 10);
    AddFaceIndex(0, 10, 11);

    // 5 adjacent faces
    AddFaceIndex(1, 5, 9);
    AddFaceIndex(5, 11, 4);
    AddFaceIndex(11, 10, 2);
    AddFaceIndex(10, 7, 6);
    AddFaceIndex(7, 1, 8);

    // 5 faces around point 3
    AddFaceIndex(3, 9, 4);
    AddFaceIndex(3, 4, 2);
    AddFaceIndex(3, 2, 6);
    AddFaceIndex(3, 6, 8);
    AddFaceIndex(3, 8, 9);

    // 5 adjacent faces
    AddFaceIndex(4, 9, 5);
    AddFaceIndex(2, 4, 11);
    AddFaceIndex(6, 2, 10);
    AddFaceIndex(8, 6, 7);
    AddFaceIndex(9, 8, 1);

    glm::ivec3 newF;
    std::vector<glm::ivec3> newFaces;

    for(i=0;i<f;i++){
        steps = FaceIndices.size();
        for(j=0;j<steps;j++){
            a = Cell3D::AddMiddlePoint(FaceIndices[j][0],FaceIndices[j][1]);
            b = Cell3D::AddMiddlePoint(FaceIndices[j][1],FaceIndices[j][2]);
            c = Cell3D::AddMiddlePoint(FaceIndices[j][2],FaceIndices[j][0]);
            newF = {FaceIndices[j][0],a,c};
            newFaces.push_back(newF);
            newF = {FaceIndices[j][1],b,a};
            newFaces.push_back(newF);
            newF = {FaceIndices[j][2],c,b};
            newFaces.push_back(newF);
            newF = {a,b,c};
            newFaces.push_back(newF);
        }
        FaceIndices = newFaces;
        newFaces.clear();
        newFaces.shrink_to_fit();
    }

    NV = (int)Verticies.size();
    ntriangles = FaceIndices.size();

    v0 = (4.0/3.0)*M_PI*pow(r0,3);
    s0 = pow((6*sqrt(M_PI)*v0*calA0),(2.0/3.0));
    a0 = (s0/(float)ntriangles);
    l0 = sqrt((4.0*a0)/sqrt(3.0));
    for(i=0;i<NV;i++){
      Verticies[i].X *= r0;
      Verticies[i].Y *= r0;
      Verticies[i].Z *= r0;
      Verticies[i].X += x0;
      Verticies[i].Y += y0;
      Verticies[i].Z += z0;
    }
    midpointCache.clear();
    midpointCache.shrink_to_fit();

  }

  void Cell3D::SetVertexPos(int vi, float x , float y, float z){
    if(vi > NV){
      std::cerr << "Vertex index is larger than the number of cell verticies" << std::endl;
    }
    else{
      Verticies[vi].X = x;
      Verticies[vi].Y = y;
      Verticies[vi].Z = z;
    }
  }

  void Cell3D::AddFaceIndex(int a, int b, int c){
    glm::ivec3 idx  = {a,b,c};
    FaceIndices.push_back(idx);
  }

  int Cell3D::AddMiddlePoint(int p1, int p2){
    int key; int i;
    if(p1 < p2)
      key = floor((p1+p2) * (p1+p2+1)/2) + p1;
    else
      key = floor((p1+p2) * (p1+p2+1)/2) + p2;
    for(i=0;i<(int)midpointCache.size();i++){
      if(key == midpointCache[i][0])
        return midpointCache[i][1];
    }

    i=Verticies.size();
    Verticies.push_back(GetMiddlePoint(p1,p2));
    std::vector<int> cache = {0,0};
    cache.shrink_to_fit();
    cache[0] = key; cache[1] = i;
    midpointCache.push_back(cache);
    return i;
  }

  Vertex3D Cell3D::GetMiddlePoint(int i, int j){
    Vertex3D p1 = Verticies[i];
    Vertex3D p2 = Verticies[j];
    Vertex3D pmid;
    pmid.X = 0.5*(p1.X + p2.X);
    pmid.Y = 0.5*(p1.Y + p2.Y);
    pmid.Z = 0.5*(p1.Z + p2.Z);
    float norm = pmid.normPos();
    pmid.X /= norm;
    pmid.Y /= norm;
    pmid.Z /= norm;
    return pmid;
  }

  float Cell3D::GetVolume(){
    int i,j;
    float x[3], y[3],z[3];
    float det, volume=0.0;
    float d1,d2,d3;
    glm::ivec3 tri;
    for(i=0;i<ntriangles;i++){
      tri = FaceIndices[i];
      for(j=0;j<3;j++){
        x[j] = Verticies[tri[j]].X;
        y[j] = Verticies[tri[j]].Y;
        z[j] = Verticies[tri[j]].Z;
      }
      d1 = x[0]*(y[1]*z[2] - y[2]*z[1]);
      d2 = -x[1]*(y[0]*z[2]- y[2]*z[0]);
      d3 = x[2]*(y[0]*z[1] - y[1]*z[0]);
      det = d1+d2+d3;
      volume += det;
    }
    volume /= 6;
    if(volume < 0.0){
      volume *= -1;
    }
    return volume;
  }

  void Cell3D::UpdateVolume(){
    Volume = GetVolume();
  }

  void Cell3D::UpdateCOM(){
    COMX = 0.0;
    COMY = 0.0;
    COMZ = 0.0;
    for(int i=0;i<NV;i++){
      COMX += Verticies[i].X;
      COMY += Verticies[i].Y;
      COMZ += Verticies[i].Z;
    }
    COMX /= NV;
    COMY /= NV;
    COMZ /= NV;
  }

}
