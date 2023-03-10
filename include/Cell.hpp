/*
 * =====================================================================================
 *
 *       Filename:  Cell.hpp
 *
 *    Description: Header file for single cudaDPM
 *
 *        Version:  1.0
 *        Created:  06/01/2022 05:22:34 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Shaka X,
 *   Organization:  Yale Univeristy
 *
 * =====================================================================================
 */

#include<vector>
#include<glm/vec3.hpp>
namespace cudaDPM{
  struct Vertex2D{
    float X; float Y;
    float Vx; float Vy;
    float Fx; float Fy;
    Vertex2D();
    Vertex2D(float x, float y);
  };
  struct Vertex3D{
    float X; float Y; float Z;
    float Vx; float Vy; float Vz;
    float Fx; float Fy; float Fz;
    Vertex3D();
    Vertex3D(float x, float y, float z);

    float normPos();
    void normalizePos();
  };
  class Cell2D{
    public:
      int NDIM;
      int NV;
      float calA0;
      float l0;
      float r0;
      float v0;
      float Ka;
      float Kb;
      float Kl;
      float Area;
      float COMX;
      float COMY;
      std::vector<int> ip1;
      std::vector<int> im1;
      float vmin;
      float Dr;
      float Ds;
      float a0;
      float psi;
      float U;
      float Ks;
      std::vector<float> l1;
      std::vector<float> l2;
      std::vector<float> radii;
      std::vector<int> NearestVertexIdx;
      std::vector<int> NearestCellIdx;
      std::vector<Vertex2D> Verticies;
      Cell2D(float x0, float y0,
             float calA, int NV, float r0,
             float ka, float kl, float kb);
      Cell2D(std::vector<Vertex2D> Verticies);

      void SetCellVelocity(float v);
      void UpdateDirectorDiffusion(float dt);
      float GetArea();
  };

  class Cell3D{
    public:
      int NDIM;
      int NV;
      float calA0;
      float r0;
      float v0;
      float a0;
      float s0;
      float U;
      float Kv;
      float Ka;
      float Ks;
      //float Kb;
      int ntriangles;
      float COMX;
      float COMY;
      float COMZ;
      float Volume;
      float SurfaceArea;
      std::vector<Vertex3D> Verticies;
      std::vector<glm::ivec3> FaceIndices;
      std::vector<std::vector<int>> midpointCache;
      int nsurfacep;
      float l0;
      Cell3D(float x0, float y0, float z0,
             float calA, int f, float r0,
             float Kv, float Ka);//, float Kb);
      Cell3D(std::vector<float> X, std::vector<float> Y, std::vector<float> Z,
             std::vector<std::vector<int>> triangles,
             float v0, float sa0,
             float Kv, float Ka);//, float Kb);
      Cell3D(std::vector<float> X ,std::vector<float> Y,std::vector<float> Z,
             float v0, float a0, float Kv, float Ka);

      void SetVertexPos(int vi, float x, float y, float z);
      void AddFaceIndex(int a, int b, int c);
      int AddMiddlePoint(int p1, int p2);
      float GetVolume();
      void UpdateVolume();
      void UpdateCOM();
      Vertex3D GetMiddlePoint(int i ,int j);
  };
}
