//
// Created by salmon on 16-6-29.
//

#include "em.h"
#include "../../../example/em/EMFluid.h"
#include "../../../example/em/PML.h"
#include "../../../src/manifold/pre_define/PreDefine.h"


namespace simpla { namespace scenario
{
typedef manifold::CartesianManifold mesh_type;

void EM::setup(ConfigParser const &options)
{

    auto mesh_center = this->add_mesh<mesh_type>();

    mesh_center->setup(options["Mesh"]).name("Center").deploy();


    this->add_problem_domain<EMFluid<mesh_type >>(mesh_center->id())
            ->setup(options).deploy();

    if (options["PML"])
    {
//        this->extend_domain<PML<mesh_type> >(mesh_center->id(), options["PML"]["Width"].as<size_type>(5), "PML_");
    }
//
//
//    {
//        std::string str = options["ProblemDomain"].as<std::string>();
//        if (str == "PIC")
//        {
//            problem_domain = std::make_shared<EMPIC < mesh_type>>
//            (&mesh_center);
//        }
//        else if (str == "Fluid")
//        {
//            problem_domain = std::make_shared<EMFluid<mesh_type>>(&mesh_center);
//
//        }
//        else
//        {
//            RUNTIME_ERROR << "Unknown problem type [" << str << "]" << std::endl;
//        }
//
//    }

}
}}//namespace simpla{namespace  scenario{