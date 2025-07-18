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

#ifndef __2DCells__
#define __2DCells__

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
}
#endif