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

#include "../../src/gtl/primitives.h"
#include "../../src/gtl/type_cast.h"
#include "../../src/io/IO.h"

#include "../../src/physics/PhysicalConstants.h"
#include "../../src/gtl/ConfigParser.h"
#include "../../src/gtl/Log.h"


#include "../../src/field/Field.h"

#include "../../src/solver/TimeDependentSolver.h"

using namespace simpla;

#include "../../src/manifold/pre_define/cylindrical.h"

#define COORDINATE_SYSTEM CylindricalCoordinate
typedef manifold::CylindricalManifold mesh_type;


using namespace simpla::task_flow;


template<typename TM>
struct Maxwell : public solver::TimeDependentSolver::Worker
{

    typedef Context context_type;

    typedef TM mesh_type,

    Maxwell(Context &ctx) :Context::Worker(ctx) { }

    virtual ~Maxwell() { }

    typedef Real scalar_type;
    typedef nTuple<scalar_type, 3> vector_type;


    Field<scalar_type, TM, mesh::EDGE> E{*this, "E0"};
    Field<scalar_type, TM, mesh::FACE> B{*this, "B0"};
    Field<vector_type, TM, mesh::VERTEX> Bv{*this, "B0v"};
    Field<scalar_type, TM, mesh::VERTEX> BB{*this, "BB"};


    virtual void work(Real dt)
    {
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
        LOG_CMD(E += (curl(B) * speed_of_light2 - J / epsilon0) * dt);
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
    };
};

template<typename TM>
struct PML : public Context::Worker
{

    typedef Context context_type;

    typedef TM mesh_type,

    Maxwell(Context &ctx) :Context::Worker(ctx) { }

    virtual ~Maxwell() { }

    typedef Real scalar_type;
    typedef nTuple<scalar_type, 3> vector_type;


    Field<scalar_type, TM, mesh::EDGE> E{*this, "E0"};
    Field<scalar_type, TM, mesh::FACE> B{*this, "B0"};
    Field<vector_type, TM, mesh::VERTEX> Bv{*this, "B0v"};
    Field<scalar_type, TM, mesh::VERTEX> BB{*this, "BB"};


    virtual void work(Real dt)
    {
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
        LOG_CMD(E += (curl(B) * speed_of_light2 - J / epsilon0) * dt);
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
        pml_solver->next_timestepE(geometry->dt(), E, B, &E);
        LOG_CMD(E -= J / epsilon0 * dt);
        pml_solver->next_timestepB(geometry->dt(), E, B, &B);
////		}
    };
};


int main(int argc, char **argv)
{
    Context ctx;

    Maxwell<mesh_type> em;

    Real dt = 1.0;

    ctx.apply(em, dt);


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

    auto J = traits::make_field<Real, EDGE>(*mesh);
    auto E = traits::make_field<Real, EDGE>(*mesh);
    auto B = traits::make_field<Real, FACE>(*mesh);

    E.clear();
    B.clear();
    J.clear();
//    E = traits::make_function_by_config<Real,EDGE>(*mesh, options["InitValue"]["E"]);

    B = traits::make_function_by_config<Real, EDGE>(*mesh, options["InitValue"]["B"]);

//    J = traits::make_function_by_config< Real,FACE>(*mesh, options["InitValue"]["J"]);

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
    io::cd("/Input/");

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
    io::cd("/Output/");

    VERBOSE << SAVE(E) << std::endl;
    VERBOSE << SAVE(B) << std::endl;
    VERBOSE << SAVE(J) << std::endl;

    LOGGER << "----------  DONE ---------- " << std::endl;

}

