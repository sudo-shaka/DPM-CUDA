/*
 * =====================================================================================
 *
 *       Filename:  GeometricFunctions.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/29/2022 04:29:29 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include<vector>
#include<glm/vec3.hpp>
#include<glm/mat3x3.hpp>

float distance(glm::vec3 a,glm::vec3 b);
float distanceSq(glm::vec3,glm::vec3 b);
glm::vec3 circumcenter(glm::vec3 a, glm::vec3 b, glm::vec3 c);
