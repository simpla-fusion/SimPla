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
#include "../../core/field/field_constraint.h"
#include "../../core/field/load_field.h"
#include "../../core/mesh/mesh.h"
#include "../../core/mesh/mesh_ids.h"
#include "../../core/mesh/structured/structured.h"

#include "../../core/model/select.h"
#include <memory>
using namespace simpla;

USE_CASE(em," Maxwell Eqs.")
{

	size_t num_of_steps = 1000;
	size_t strides = 10;
	Real dt = 0.001;

	if (options["SHOW_HELP"])
	{
		SHOW_OPTIONS("-n,--number_of_steps <NUMBER_OF_STEPS>",
				"number of steps = <NUMBER_OF_STEPS> ,default="
						+ value_to_string(num_of_steps));
		SHOW_OPTIONS("-s,--strides <STRIDES>",
				" dump record per <STRIDES> steps, default="
						+ value_to_string(strides));
		return;
	}

	options["n"].as(&num_of_steps);

	options["s"].as<size_t>(&strides);

	auto mesh = std::make_shared<CartesianRectMesh>();

	mesh->dimensions(
			options["dimensions"].as(nTuple<size_t, 3>( { 10, 10, 10 })));

	mesh->extents(options["xmin"].as(nTuple<Real, 3>( { 0, 0, 0 })),
			options["xmax"].as(nTuple<Real, 3>( { 1, 1, 1 })));

	mesh->dt(options["dt"].as<Real>(1.0));

	mesh->deploy();

	if (GLOBAL_COMM.process_num()==0)
	{

		MESSAGE << endl

		<< "[ Configuration ]" << endl

		<< " Description=\"" << options["Description"].as<std::string>("") << "\""
		<< endl

		<< " Mesh =" << endl << "  {" << *mesh << "} " << endl

		<< " TIME_STEPS = " << num_of_steps << endl;
	}
	// Load initialize value
	auto phi = make_form<VERTEX, Real>(mesh);
	auto J = make_form<EDGE, Real>(mesh);
	auto E = make_form<EDGE, Real>(mesh);
	auto B = make_form<FACE, Real>(mesh);

	J.clear();
	E.clear();
	B.clear();
	phi.clear();

	VERBOSE_CMD(load_field(options["InitValue"]["phi"], &phi));
	VERBOSE_CMD(load_field(options["InitValue"]["B"], &B));
	VERBOSE_CMD(load_field(options["InitValue"]["E"], &E));
	VERBOSE_CMD(load_field(options["InitValue"]["J"], &J));

	auto J_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["J"]);

	auto B_src = make_field_function_by_config<FACE, Real>(*mesh,
			options["Constraint"]["B"]);

	auto E_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["E"]);

////	auto J_src = make_constraint<decltype(J)>(J.mesh(),
////			options["Constraint"]["J"]);
////	auto B_src = make_constraint<decltype(B)>(B.mesh(),
////			options["Constraint"]["B"]);
//
	LOGGER << "----------  Dump input ---------- " << endl;

	cd("/Input/");

	VERBOSE << SAVE(phi) << endl;
	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;
	VERBOSE << SAVE(J) << endl;

	{
		LOGGER << "----------  START ---------- " << endl;

		cd("/Save/");
		for (size_t step = 0; step < num_of_steps; ++step)
		{
			VERBOSE << "Step [" << step << "/" << num_of_steps << "]" << endl;

			E += E_src;
//			J += J_src;
//			B += B_src;
			E = curl(B) * dt - J;
			B = -curl(E) * dt;

			VERBOSE << SAVE_APPEND(E) << endl;
			VERBOSE << SAVE_APPEND(B) << endl;

		}
	}
	cd("/Output/");
	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;
	VERBOSE << SAVE(J) << endl;

	LOGGER << "----------  DONE ---------- " << endl;

}

