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

#include "../../core/field/field.h"

#include "../../core/mesh/mesh.h"

#include "../../core/model/select.h"

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

	mesh->dimensions(
			options["dimensions"].as(nTuple<size_t, 3>( { 10, 10, 10 })));

	mesh->extents(options["xmin"].as(nTuple<Real, 3>( { 0, 0, 0 })),
			options["xmax"].as(nTuple<Real, 3>( { 1, 1, 1 })));

	mesh->dt(options["dt"].as<Real>(1.0));

	mesh->deploy();

	SHOW(GLOBAL_COMM.get_coordinate(GLOBAL_COMM.process_num()));

	if (GLOBAL_COMM.process_num()==0)
	{

		MESSAGE << endl

		<< "[ Configuration ]" << endl

		<< " Description=\"" << options["Description"].as<std::string>("") << "\""
		<< endl

		<< " Mesh =" << endl << "  {" << *mesh << "} " << endl

		<< " TIME_STEPS = " << num_of_steps << endl;
	}

	std::shared_ptr<PML<mesh_type>> pml_solver;

	if (options["FieldSolver"]["PML"])
	{
		pml_solver = std::make_shared<PML<mesh_type>>(*mesh,
				options["FieldSolver"]["PML"]);

	}

	auto J = mesh->template make_form<EDGE, Real>();
	auto E = mesh->template make_form<EDGE, Real>();
	auto B = mesh->template make_form<FACE, Real>();

	J.clear();
	E.clear();
	B.clear();

	VERBOSE_CMD(load_field(options["InitValue"]["B"], &B));
	VERBOSE_CMD(load_field(options["InitValue"]["E"], &E));
	VERBOSE_CMD(load_field(options["InitValue"]["J"], &J));

	auto J_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["J"]);

	auto B_src = make_field_function_by_config<FACE, Real>(*mesh,
			options["Constraint"]["B"]);

	auto E_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["E"]);

	LOGGER << "----------  Dump input ---------- " << endl;

	E.wait();
	B.wait();
	J.wait();

	cd("/Input/");

	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;
	VERBOSE << SAVE(J) << endl;

	LOGGER << "----------  START ---------- " << endl;

	DEFINE_PHYSICAL_CONST

	Real dt = mesh->dt();

	cd("/Save/");

	for (size_t step = 0; step < num_of_steps; ++step)
	{
		VERBOSE << "Step [" << step << "/" << num_of_steps << "]" << endl;

		J = J_src;
		J.wait();
//			J += J_src;
//			B += B_src;

		if (!pml_solver)
		{
			LOG_CMD(E += (curl(B) / mu0 - J) / epsilon0 * dt);
			LOG_CMD(B += -curl(E) * dt);
		}
		else
		{
			pml_solver->next_timestepE(mesh->dt(), E, B, &E);
			LOG_CMD(E -= J / epsilon0 * dt);
			pml_solver->next_timestepB(mesh->dt(), E, B, &B);
		}

		B.wait();
		E.wait();

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
