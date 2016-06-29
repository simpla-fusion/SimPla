//
// Created by salmon on 16-6-29.
//

#include "em.h"
#include "../problem_domain/EMFluid.h"
#include "../problem_domain/PML.h"

namespace simpla { namespace scenario
{
virtual void EM::setup()
{

    auto mesh_center = ctx.add_mesh<mesh_type>();

    mesh_center->setup(options["Mesh"]).name("Center").deploy();


    ctx.add_problem_domain<EMFluid<mesh_type >>(mesh_center->id())
            ->setup(options).deploy();

    if (options["PML"])
    {
        ctx.extend_domain<PML<mesh_type> >(mesh_center->id(), options["PML"]["Width"].as<size_type>(5), "PML_");
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