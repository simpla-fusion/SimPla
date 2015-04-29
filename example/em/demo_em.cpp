/*
 * @file demo_em.cpp
 *
 *  Created on: 2014年11月28日
 *      Author: salmon
 */

#include <stddef.h>
#include <iostream>
#include <string>

#include "../../core/application/application.h"
#include "../../core/application/use_case.h"

#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"
#include "../../core/physics/physical_constants.h"
#include "../../core/field/field.h"

#include "../../core/mesh/mesh.h"

#include "../../applications/field_solver/pml.h"

#include <memory>
using namespace simpla;

typedef CartesianRectMesh mesh_type;

USE_CASE(em," Maxwell Eqs.")
{

	size_t num_of_steps = 1000;
	size_t strides = 10;

	if (options["case_help"])
	{

		MESSAGE

		<< " Options:" << endl

				<<

				"\t -n,\t--number_of_steps <NUMBER>  \t, Number of steps = <NUMBER> ,default="
						+ value_to_string(num_of_steps)
						+ "\n"
								"\t -s,\t--strides <NUMBER>            \t, Dump record per <NUMBER> steps, default="
						+ value_to_string(strides) + "\n";

		return;
	}

	options["n"].as(&num_of_steps);

	options["s"].as<size_t>(&strides);

	auto mesh = std::make_shared<mesh_type>();

	mesh->load(options["Mesh"]);

	mesh->deploy();

//	typename mesh_type::coordinates_type zero = { 0.5, 0, 0 };
//
//	auto ss = mesh->template coordinates_to_id<VERTEX>(zero);
//	SHOW(mesh->template coordinates_to_id<VERTEX>(zero));
//	SHOW(mesh->template coordinates_to_index<VERTEX>(zero));
//	SHOW(mesh->template coordinates_to_id_tuple<VERTEX>(zero));
//	SHOW(mesh->template unpack_index<VERTEX>(ss));
//	SHOW(mesh->node_id(ss));
//	SHOW(mesh->node_id(ss + (mesh_type::_DI)));
//	SHOW(mesh->node_id(ss + (mesh_type::_DI << 1)));
//
//	CHECK_BIT(mesh->_D);
//	CHECK_BIT(mesh->ID_MASK);
//	CHECK_BIT(mesh->ID_ZERO);
//	CHECK_BIT(mesh->CARRAY_FLAG);
//	CHECK_BIT(mesh->SUB_ID_MASK);
//	CHECK_BIT(mesh->PRIMARY_ID_MASK);
//
//	nTuple<long, 3> idx = { 1, 2, 3 };
//	ss = (mesh->template pack_index<VERTEX>(idx));
//
//	CHECK_BIT(ss);
//	CHECK_BIT(mesh->INDEX_ZERO);
//	SHOW(idx);
//	SHOW(mesh->template unpack_index<VERTEX>(ss));
//
//	auto s = mesh->coordinates_to_id_tuple(zero);

	if (GLOBAL_COMM.process_num()==0)
	{

		MESSAGE << endl

		<< "[ Configuration ]" << endl

		<< " Description=\"" << options["Description"].as<std::string>("") << "\""
		<< endl

		<< " Mesh =" << endl << "  {" << *mesh << "} " << endl

		<< " TIME_STEPS = " << num_of_steps << endl;
	}

//	std::shared_ptr<PML<mesh_type>> pml_solver;
//
//	if (options["FieldSolver"]["PML"])
//	{
//		pml_solver = std::make_shared<PML<mesh_type>>(*mesh,
//				options["FieldSolver"]["PML"]);
//
//	}

	auto J = mesh->template make_form<EDGE, Real>();
	auto E = mesh->template make_form<EDGE, Real>();
	auto B = mesh->template make_form<FACE, Real>();

	VERBOSE_CMD(load_field(options["InitValue"]["E"], &E));
	VERBOSE_CMD(load_field(options["InitValue"]["J"], &J));
	VERBOSE_CMD(load_field(options["InitValue"]["B"], &B));

	auto J_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["J"]);

	auto E_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["E"]);

//	auto E_Boundary = E;
//	auto B_Boundary = B;
//
//	if (options["PEC"])
//	{
//		filter_domain_by_config(options["PEC"]["Domain"], &B_Boundary.domain());
//		filter_domain_by_config(options["PEC"]["Domain"], &E_Boundary.domain());
//	}
//	else
//	{
//		E_Boundary.clear();
//		B_Boundary.clear();
//	}

	LOGGER << "----------  Dump input ---------- " << endl;

	cd("/Input/");

	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;
	VERBOSE << SAVE(J) << endl;

	DEFINE_PHYSICAL_CONST
	Real dt = mesh->dt();
	auto dx = mesh->dx();

	Real omega = 0.01 * PI / dt;

	LOGGER << "----------  START ---------- " << endl;

	cd("/Save/");

	for (size_t step = 0; step < num_of_steps; ++step)
	{
		VERBOSE << "Step [" << step << "/" << num_of_steps << "]" << endl;

		J.self_assign(J_src);
		E.self_assign(E_src);

//		if (!pml_solver)
		{

			LOG_CMD(E += curl(B) * (dt * speed_of_light2) - J * (dt / epsilon0));
//			E_Boundary = 0;
//			B_Boundary = 0;
			LOG_CMD(B -= curl(E) * dt);

		}
//		else
//		{
//			pml_solver->next_timestepE(mesh->dt(), E, B, &E);
//			LOG_CMD(E -= J / epsilon0 * dt);
//			pml_solver->next_timestepB(mesh->dt(), E, B, &B);
//		}

		VERBOSE << SAVE_RECORD(J) << endl;
		VERBOSE << SAVE_RECORD(E) << endl;
		VERBOSE << SAVE_RECORD(B) << endl;

		mesh->next_timestep();

	}

	cd("/Output/");
	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;
	VERBOSE << SAVE(J) << endl;

	LOGGER << "----------  DONE ---------- " << endl;

}

//	auto phi = make_form<VERTEX, Real>(mesh);
//
//	phi.clear();
//
//	int N = GLOBAL_COMM.process_num();
//
//	int count = 0;
//
//	phi.for_each([&](Real &v)
//	{
//		v=(N+1)*100 +count;
//		++count;
//	});
//
////	phi = N;
//
//	GLOBAL_COMM.barrier();
//	if (GLOBAL_COMM.process_num()==0)
//	{
//		VERBOSE<<std::endl<<phi<<endl;
//	}
//	GLOBAL_COMM.barrier();
//	if (GLOBAL_COMM.process_num()==1)
//	{
//		VERBOSE<<std::endl<<phi<<endl;
//	}
//	GLOBAL_COMM.barrier();
//
//	phi.sync();
//	phi.wait();
//	if (GLOBAL_COMM.process_num()==0)
//	{
//		VERBOSE<<std::endl<<phi<<endl;
//	}
//	GLOBAL_COMM.barrier();
//	if (GLOBAL_COMM.process_num()==1)
//	{
//		VERBOSE<<std::endl<<phi<<endl;
//	}
//	GLOBAL_COMM.barrier();
//
//	VERBOSE << SAVE(phi) << endl;
