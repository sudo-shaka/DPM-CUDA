/*
 * =====================================================================================
 *
 *       Filename:  GeometricFunctions.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/29/2022 04:13:33 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/gtx/norm.hpp>

float distance(glm::vec3 a, glm::vec3 b){
  glm::vec3 tmp = a-b;
  return sqrt(glm::dot(tmp,tmp));
}

float distanceSq(glm::vec3 a, glm::vec3 b){
  glm::vec3 temp = a-b;
  return glm::dot(temp,temp);
}

glm::vec3 circumcenter(glm::vec3 a, glm::vec3 b, glm::vec3 c){
  glm::vec3 ac = c-a;
  glm::vec3 ab = b-a;
  glm::vec3 abXac = glm::cross(ab,ac);
  glm::vec3 toCircumSprCenter = glm::cross(abXac,ab)*glm::length2(ac) + glm::cross(ac,abXac)*glm::length2(ab) /
    (2.0f*glm::length2(abXac));
  return a + toCircumSprCenter;
}

