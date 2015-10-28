/*
 * @file demo_em.cpp
 *
 *  Created on: 2014-11-28
 *      Author: salmon
 */

#include <stddef.h>
#include <iostream>
#include <memory>
#include <string>

#include "../../core/application/application.h"
#include "../../core/gtl/primitives.h"
#include "../../core/gtl/type_cast.h"
#include "../../core/io/io.h"

#include "../../core/physics/constants.h"
#include "../../core/physics/physical_constants.h"
#include "../../core/gtl/utilities/config_parser.h"
#include "../../core/gtl/utilities/log.h"


#include "../../core/manifold/manifold_traits.h"
#include "../../core/manifold/calculus.h"
#include "../../core/manifold/domain.h"

#include "../../core/field/field.h"

using namespace simpla;

#ifdef CYLINDRICAL_COORDINATE_SYTEM
#include "../../core/manifold/pre_define/cylindrical.h"
#define COORDINATE_SYSTEM CylindricalCoordinate
typedef manifold::CylindricalCoRect mesh_type;
#else

#include "../../core/manifold/pre_define/cartesian.h"

#define COORDINATE_SYSTEM CartesianCoordinate<3>
typedef manifold::Cartesian<3> mesh_type;

#endif


SP_APP(em, " Maxwell Eqs.")
{

    size_t num_of_steps = 1000;
    size_t check_point = 10;

    if (options["case_help"])
    {
        MESSAGE

        << " Options:" << std::endl

        << "\t -n,\t--number_of_steps <NUMBER>  \t, Number of steps = <NUMBER> ,default="

        << type_cast<std::string>(num_of_steps) << std::endl

        << "\t -s,\t--strides <NUMBER>            \t, Dump record per <NUMBER> steps, default="

        << type_cast<std::string>(check_point) << std::endl;

        return;
    }

    num_of_steps = options["n"].as<size_t>(num_of_steps);

    check_point = options["check_point"].as<size_t>(check_point);

    auto mesh = std::make_shared<mesh_type>();

    mesh->load(options);

    mesh->deploy();

    MESSAGE << std::endl

    << "[ Configuration ]" << std::endl

    << "Description=\"" << options["Description"].as<std::string>("") << "\"" << std::endl

    << *mesh << std::endl

    << " TIME_STEPS = " << num_of_steps << std::endl;

//	std::shared_ptr<PML < mesh_type>>	pml_solver;
//
//	if (options["FieldSolver"]["PML"])
//	{
//		pml_solver = std::make_shared<PML < mesh_type>>
//		(*geometry,
//				options["FieldSolver"]["PML"]);
//
//	}

    auto J = traits::make_field<EDGE, Real>(*mesh);

    auto E = traits::make_field<EDGE, Real>(*mesh);

    auto B = traits::make_field<FACE, Real>(*mesh);

    E.clear();
    B.clear();
    J.clear();
//    E = traits::make_function_by_config<EDGE, Real>(*mesh, options["InitValue"]["E"]);

    B = traits::make_function_by_config<EDGE, Real>(*mesh, options["InitValue"]["B"]);

//    J = traits::make_function_by_config<FACE, Real>(*mesh, options["InitValue"]["J"]);

//    auto J_src = traits::make_function_by_config<Real>(options["Constraint"]["J"], traits::make_domain<EDGE>(*mesh));
//
//    auto E_src = traits::make_function_by_config<Real>(options["Constraint"]["E"], traits::make_domain<EDGE>(*mesh));
//
//    auto E_Boundary = E;
//
//    auto B_Boundary = B;
//
//    if (options["PEC"])
//    {
//        filter_by_config(options["PEC"]["Domain"], &B_Boundary.domain());
//        filter_by_config(options["PEC"]["Domain"], &E_Boundary.domain());
//    }
//    else
//    {
//        E_Boundary.clear();
//        B_Boundary.clear();
//    }
//
//    LOGGER << "----------  Dump input ---------- " << std::endl;
//
//    cd("/Input/");

    VERBOSE << SAVE(E) << std::endl;
    VERBOSE << SAVE(B) << std::endl;
    VERBOSE << SAVE(J) << std::endl;

    DEFINE_PHYSICAL_CONST

    Real dt = mesh->dt();

//    Real omega = 0.01 * PI / dt;
//
//    LOGGER << "----------  START ---------- " << std::endl;
//
//    cd("/Save/");
//
//    for (size_t step = 0; step < num_of_steps; ++step)
//    {
//        VERBOSE << "Step [" << step << "/" << num_of_steps << "]" << std::endl;
//
////        J.self_assign(J_src);
////        E.self_assign(E_src);
//
//        J = J_src;
//
//        E = E_src;
//
////		if (!pml_solver)
////        {
//
//        LOG_CMD(E += (curl(B) * speed_of_light2 - J / epsilon0) * dt);
//
//        E_Boundary = 0;
//
    LOG_CMD(B -= curl(E) * dt);
//
//        B_Boundary = 0;
//
////        }
////		else
////		{
////			pml_solver->next_timestepE(geometry->dt(), E, B, &E);
////			LOG_CMD(E -= J / epsilon0 * dt);
////			pml_solver->next_timestepB(geometry->dt(), E, B, &B);
////		}
//
//
//        mesh->next_time_step();
//
//        if (step % check_point == 0)
//        {
//            VERBOSE << SAVE_RECORD(J) << std::endl;
//            VERBOSE << SAVE_RECORD(E) << std::endl;
//            VERBOSE << SAVE_RECORD(B) << std::endl;
//        }
//
//    }
//
//    cd("/Output/");
//
//    VERBOSE << SAVE(E) << std::endl;
//    VERBOSE << SAVE(B) << std::endl;
//    VERBOSE << SAVE(J) << std::endl;

    LOGGER << "----------  DONE ---------- " << std::endl;

}

