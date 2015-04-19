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
#include "../../core/mesh/mesh.h"
#include "../../core/mesh/structured/structured.h"
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

	auto mesh = std::make_shared<CartesianManifold>();

	mesh->dimensions(
			options["dimensions"].as(nTuple<size_t, 3>( { 10, 10, 10 })));

	mesh->extents(options["xmin"].as(nTuple<Real, 3>( { 0, 0, 0 })),
			options["xmax"].as(nTuple<Real, 3>( { 1, 1, 1 })));

	mesh->dt(options["dt"].as<Real>(1.0));

	mesh->deploy();

	LOGGER << std::endl

	<< "[ Configuration ]" << std::endl

	<< " Description=\"" << options["Description"].as<std::string>("") << "\""
			<< std::endl

			<< " Mesh =" << std::endl << "  {" << *mesh << "} " << std::endl

			<< " TIME_STEPS = " << num_of_steps << std::endl

			;

	// Load initialize value

	auto phi = make_form<VERTEX, Real>(mesh);

	auto J = make_form<EDGE, Real>(mesh);
	auto E = make_form<EDGE, Real>(mesh);
	auto B = make_form<FACE, Real>(mesh);

//	VERBOSE_CMD(B = load_field < EDGE, Real > (mesh, options["InitValue"]["B"]))
	;
//	VERBOSE_CMD(apply_constraint(options["InitValue"]["E"], &E));
//	VERBOSE_CMD(apply_constraint(options["InitValue"]["J"], &J));

//	auto E_src = make_constraint<decltype(E)>(E.mesh(),
//			options["Constraint"]["E"]);
//	auto J_src = make_constraint<decltype(J)>(J.mesh(),
//			options["Constraint"]["J"]);
//	auto B_src = make_constraint<decltype(B)>(B.mesh(),
//			options["Constraint"]["B"]);

	LOGGER << "----------  Dump input ---------- " << std::endl;

	E.clear();

	E = 1.234;

	cd("/Input/");

	VERBOSE << SAVE(E) << std::endl;
////	VERBOSE << SAVE(E) << std::endl;
////	VERBOSE << SAVE(J) << std::endl;
//
//	if (options["JUST_A_TEST"])
//	{
//		LOGGER << " Just test configuration!" << std::endl;
//	}
//	else
//	{
//		LOGGER << "----------  START ---------- " << std::endl;
//
//		cd("/Save/");
//		for (size_t s = 0; s < num_of_steps; ++s)
//		{
//			VERBOSE << "Step [" << s << "/" << num_of_steps << "]" << std::endl;
//
////			E_src(&E);
////			J_src(&J);
////			B_src(&B);
//			E = curl(B) * dt - J;
//			B = -curl(E) * dt;
//
//			VERBOSE << SAVE_RECORD(E) << std::endl;
//			VERBOSE << SAVE_RECORD(B) << std::endl;
//
//		}
//	}
//	cd("/Output/");
////	VERBOSE << SAVE(E) << std::endl;
////	VERBOSE << SAVE(B) << std::endl;
////	VERBOSE << SAVE(J) << std::endl;

	LOGGER << "----------  DONE ---------- " << std::endl;

}

