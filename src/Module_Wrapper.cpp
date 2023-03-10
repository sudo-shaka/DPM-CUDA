/*
 * =====================================================================================
 *
 *       Filename:  cudaDPM_Wrapper.cpp
 *
 *    Description: Wrapper for python module
 *
 *        Version:  1.0
 *        Created:  06/02/2022 10:48:51 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shaka X
 *   Organization:  Yale University
 *
 * =====================================================================================
 */

#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include"../include/Cell.hpp"
#include"../include/Tissue.hpp"
#include"../include/DPMCudaKernel.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_Vertex2D(py::module &);
void init_Cell2D(py::module &);
void init_Tissue2D(py::module &);
void init_Cell3D(py::module &);
void init_Tissue3D(py::module &);
void init_Vertex3D(py::module &);

void init_vec3(py::module &);
void init_ivec3(py::module &);

void init_vec3(py::module &m){
    py::class_<glm::vec3>(m,"vec3")
    .def(py::init<>())
    .def(py::init<float,float,float>(),py::arg("x"),py::arg("y"),py::arg("z"))
    .def_readwrite("x",&glm::vec3::x)
    .def_readwrite("y",&glm::vec3::y)
    .def_readwrite("z",&glm::vec3::z)
    ;
}

void init_ivec3(py::module &m){
    py::class_<glm::ivec3>(m,"ivec3")
    .def(py::init<>())
    .def(py::init<float,float,float>(),py::arg("x"),py::arg("y"),py::arg("z"))
    .def_readwrite("x",&glm::ivec3::x)
    .def_readwrite("y",&glm::ivec3::y)
    .def_readwrite("z",&glm::ivec3::z)
    ;
}

void init_Vertex2D(py::module &m){
  py::class_<cudaDPM::Vertex2D>(m,"Vertex2D")
    .def(py::init<>())
    .def(py::init<float, float>(),py::arg("x"),py::arg("y"))
    .def_readwrite("x",&cudaDPM::Vertex2D::X)
    .def_readwrite("y",&cudaDPM::Vertex2D::Y)
    .def_readwrite("vx",&cudaDPM::Vertex2D::Vx)
    .def_readwrite("vy",&cudaDPM::Vertex2D::Vy)
    .def_readwrite("fx",&cudaDPM::Vertex2D::Fx)
    .def_readwrite("fy",&cudaDPM::Vertex2D::Fy)
    ;
}

void init_Cell2D(py::module &m){
  py::class_<cudaDPM::Cell2D>(m,"Cell2D")
    .def(py::init<std::vector<cudaDPM::Vertex2D>>(),py::arg("VertexArray"))
    .def(py::init<float, float, float, int, float, float, float, float>(),
        py::arg("x0"),py::arg("y0"),
        py::arg("CalA0"),py::arg("NV"),py::arg("r0"),
        py::arg("Ka"),py::arg("Kl"),py::arg("Kb")
    )
    .def_readonly("Verticies",&cudaDPM::Cell2D::Verticies)
    .def_readonly("NV",&cudaDPM::Cell2D::NV)
    .def_readonly("CalA0",&cudaDPM::Cell2D::calA0)
    .def_readonly("a0",&cudaDPM::Cell2D::a0)
    .def_readonly("l0",&cudaDPM::Cell2D::l0)
    .def_readonly("r0",&cudaDPM::Cell2D::r0)
    .def_readwrite("Ka",&cudaDPM::Cell2D::Ka)
    .def_readwrite("Kb",&cudaDPM::Cell2D::Kb)
    .def_readwrite("Kl",&cudaDPM::Cell2D::Kl)
    .def_readwrite("Ks",&cudaDPM::Cell2D::Ks)
    .def_readwrite("vmin",&cudaDPM::Cell2D::vmin)
    .def_readwrite("Dr",&cudaDPM::Cell2D::Dr)
    .def_readwrite("Ds",&cudaDPM::Cell2D::Ds)
    .def_readwrite("psi",&cudaDPM::Cell2D::psi)
    .def_readonly("U",&cudaDPM::Cell2D::U)
    .def("GetArea",&cudaDPM::Cell2D::GetArea)
    .def("SetVelocity",&cudaDPM::Cell2D::SetCellVelocity)
    .def("UpdateDirectorDiffusion",&cudaDPM::Cell2D::UpdateDirectorDiffusion)
    ;
}

void init_Tissue2D(py::module &m){
  py::class_<cudaDPM::Tissue2D>(m,"Tissue2D")
  .def(py::init<std::vector<cudaDPM::Cell2D>,float>(),
  py::arg("Cells"),py::arg("phi0")
  )
  .def_readonly("NCELLS",&cudaDPM::Tissue2D::NCELLS)
  .def_readonly("phi0",&cudaDPM::Tissue2D::phi0)
  .def_readwrite("Cells",&cudaDPM::Tissue2D::Cells)
  .def_readwrite("Kc",&cudaDPM::Tissue2D::Kc)
  .def_readonly("BoxLength",&cudaDPM::Tissue2D::L)
  .def_readonly("VertDOF",&cudaDPM::Tissue2D::VertDOF)
  .def("EulerUpdate",&cudaDPM::Tissue2D::EulerUpdate)
  .def("disperse",&cudaDPM::Tissue2D::disperse)
  ;
}

void init_Vertex3D(py::module &m){
  py::class_<cudaDPM::Vertex3D>(m,"Vertex3D")
    .def(py::init<>())
    .def(py::init<float,float,float>(),
         py::arg("X"),
         py::arg("Y"),
         py::arg("Z")
         )
    .def_readwrite("x",&cudaDPM::Vertex3D::X)
    .def_readwrite("y",&cudaDPM::Vertex3D::Y)
    .def_readwrite("z",&cudaDPM::Vertex3D::Z)
    .def_readwrite("vx",&cudaDPM::Vertex3D::Vx)
    .def_readwrite("vy",&cudaDPM::Vertex3D::Vy)
    .def_readwrite("vz",&cudaDPM::Vertex3D::Vz)
    .def_readwrite("fx",&cudaDPM::Vertex3D::Fx)
    .def_readwrite("fy",&cudaDPM::Vertex3D::Fy)
    .def_readwrite("fz",&cudaDPM::Vertex3D::Fz)
    ;
}

void init_Cell3D(py::module &m){
  py::class_<cudaDPM::Cell3D>(m,"Cell3D")
    .def(py::init<float,float,float,float,int, float, float, float>(),
      py::arg("x0"),
      py::arg("y0"),
      py::arg("z0"),
      py::arg("CalA0"),
      py::arg("VertexRecursion"),
      py::arg("r0"),
      py::arg("Kv"),
      py::arg("Ka")
    )
    .def(py::init<std::vector<float>,
                  std::vector<float>,
                  std::vector<float>,
                  std::vector<std::vector<int>>,
                  float, float, float, float>(),
      py::arg("Xvals"),
      py::arg("Yvals"),
      py::arg("Zvals"),
      py::arg("TriangleList"),
      py::arg("v0"),
      py::arg("sa0"),
      py::arg("Kv"),
      py::arg("Ka")
    )
    .def_readonly("NV",&cudaDPM::Cell3D::NV)
    .def_readonly("NT",&cudaDPM::Cell3D::ntriangles)
    .def_readwrite("Ks",&cudaDPM::Cell3D::Ks)
    .def_readonly("v0",&cudaDPM::Cell3D::v0)
    .def_readonly("sa0",&cudaDPM::Cell3D::a0)
    .def_readwrite("Kv",&cudaDPM::Cell3D::Kv)
    .def_readwrite("Ka",&cudaDPM::Cell3D::Ka)
    .def_readwrite("Kb",&cudaDPM::Cell3D::Kv)
    .def_readwrite("Verticies",&cudaDPM::Cell3D::Verticies)
    .def_readwrite("FaceIndices",&cudaDPM::Cell3D::FaceIndices)
    .def("GetVolume",&cudaDPM::Cell3D::GetVolume)
    ;
}

void init_Tissue3D(py::module &m){
  py::class_<cudaDPM::Tissue3D>(m, "Tissue3D")
    .def(py::init<std::vector<cudaDPM::Cell3D>, float>(),
         py::arg("Cells"),
         py::arg("phi0")
         )
    .def_readwrite("Kc",&cudaDPM::Tissue3D::Kc)
    .def_readwrite("Cells",&cudaDPM::Tissue3D::Cells)
    .def_readonly("BoxLength",&cudaDPM::Tissue3D::L)
    .def_readonly("NCELLS",&cudaDPM::Tissue3D::NCELLS)
    .def("EulerUpdate",&cudaDPM::Tissue3D::EulerUpdate)
    .def("disperse3D",&cudaDPM::Tissue3D::disperse3D)
    .def("disperse2D",&cudaDPM::Tissue3D::disperse2D)
    ;
}
namespace dpm {
  PYBIND11_MODULE(cudaDPM, m){
    m.doc() = "cudaDPM";
    init_Cell2D(m);
    init_Tissue2D(m);
    init_Vertex2D(m);

    init_Cell3D(m);
    init_Tissue3D(m);
    init_Vertex3D(m);

    init_ivec3(m);
  }
}

