/*
 * @file demo_pic.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "demo_pic.h"

#include <stddef.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"
#include "../../core/application/application.h"
#include "../../core/application/use_case.h"
#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/mesh/mesh.h"
#include "../../core/particle/particle.h"

using namespace simpla;
typedef CartesianRectMesh mesh_type;

USE_CASE(pic," Particle in cell" )
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

	mesh->dimensions(options["dimensions"].template as(nTuple<size_t, 3>( { 10,
			10, 10 })));

	mesh->extents(options["xmin"].template as(nTuple<Real, 3>( { 0, 0, 0 })),
			options["xmax"].template as(nTuple<Real, 3>( { 1, 1, 1 })));

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

	typedef PICDemo engine_type;

	size_t pic = 10;

	options["pic"].as(&pic);

	MESSAGE << "======== Initialize ========" << std::endl;

	auto ion = make_kinetic_particle<engine_type>(
			mesh->template domain<VOLUME>());

	ion->mass(1.0);

	ion->charge(2.0);

	ion->temperature(3.0);

	ion->deploy();

	auto extents = mesh->extents();

	auto p_generator = simple_particle_generator(*ion, extents, 1.0);

	std::mt19937 rnd_gen;

	for (size_t i = 0, ie = pic * ion->domain().size(); i < ie; ++i)
	{
		ion->insert(p_generator(rnd_gen));
	}

//	VERBOSE << save("H0", ion->dataset()) << std::endl;
//
//	ion->sync();
//	ion->wait();
	VERBOSE << save("H1", ion->dataset()) << std::endl;
//
//	ion->sync();
//	ion->wait();
//	ion->rehash();
//
//	VERBOSE << save("H2", ion->dataset()) << std::endl;
//
// Load initialize value
//	auto J = make_form<EDGE, Real>(mesh);
//	auto E = make_form<EDGE, Real>(mesh);
//	auto B = make_form<FACE, Real>(mesh);
//
//	auto E_src = make_constraint<EDGE, Real>(mesh, options["Constraint"]["E"]);
//	auto J_src = make_constraint<EDGE, Real>(mesh, options["Constraint"]["J"]);
//	auto B_src = make_constraint<FACE, Real>(mesh, options["Constraint"]["B"]);
//
//	VERBOSE_CMD(load(options["InitValue"]["B"], &B));
//	VERBOSE_CMD(load(options["InitValue"]["E"], &E));
//	VERBOSE_CMD(load(options["InitValue"]["J"], &J));

	MESSAGE << "======== START! ========" << std::endl;

//	cd("/Input/");
//
////	VERBOSE << SAVE(E);
////	VERBOSE << SAVE(B);
////	VERBOSE << SAVE(J);
//
//	cd("/Save/");
//
//	for (size_t s = 0; s < num_of_steps; ++s)
//	{
//
////		E_src(&E);
////		B_src(&B);
////
////		J.clear();
////		ion->next_timestep(dt, E, B, &J);
////		J_src(&J);
////
////		E += curl(B) * dt - J;
////		B += -curl(E) * dt;
////
////		if (s % strides == 0)
////		{
////			VERBOSE << save("H", *ion, SP_APPEND);
////			VERBOSE << save("E", E, SP_APPEND);
////			VERBOSE << save("B", B, SP_APPEND);
////
////		}
//	}
//	cd("/Output/");
//	VERBOSE << SAVE(E) << std::endl;
//	VERBOSE << SAVE(B) << std::endl;
//	VERBOSE << SAVE(J) << std::endl;

//	VERBOSE << save("H", ion->dataset()) << std::endl;

	MESSAGE << "======== DONE! ========" << std::endl;

}
