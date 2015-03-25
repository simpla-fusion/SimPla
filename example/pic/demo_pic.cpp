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
#include "../../core/particle.h"

using namespace simpla;

USE_CASE(pic)
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
		SHOW_OPTIONS("-dt  <DT>",
				" value of time step,default =" + value_to_string(dt));

		return;
	}

	options["NUMBER_OF_STEPS"].as(&num_of_steps);

	options["STRIDES"].as<size_t>(&strides);

	typedef CartesianCoordinates<RectMesh> mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::index_tuple index_tuple;

	typedef PICDemo engine_type;

	size_t pic = 10;
	index_tuple dims = { 10, 10, 10 };
	index_tuple ghost_width = { 0, 0, 0 };
	coordinates_type xmin = { 0, 0, 0 };
	coordinates_type xmax = { 1, 1, 1 };

	options["dims"].as(&dims);

	options["gw"].as(&ghost_width);

	options["pic"].as(&pic);

	auto mesh = make_mesh<mesh_type>();

	mesh->dimensions(dims);
	mesh->extents(xmin, xmax);
	mesh->ghost_width(ghost_width);
	mesh->dt(dt);
	mesh->deploy();

	MESSAGE << std::endl;

	MESSAGE << "======== Configuration ========" << std::endl;
	MESSAGE << " Description:"
			<< options["Description"].template as<std::string>("") << std::endl;

	MESSAGE << " Options:" << std::endl;

	RIGHT_COLUMN(" mesh" ) << " = {" << *mesh << "}," << std::endl;

	RIGHT_COLUMN(" time_step" ) << " = " << num_of_steps << std::endl;

	MESSAGE << "======== Initialize ========" << std::endl;

	auto ion = make_kinetic_particle<engine_type>(*mesh);

	ion->mass(1.0);

	ion->charge(2.0);

	ion->temperature(3.0);

	ion->deploy();

	auto extents = mesh->local_extents();

	auto range = mesh->local_range();

	auto p_generator = simple_particle_generator(*ion, extents, 1.0);

	std::mt19937 rnd_gen;

	for (int i = 0, ie = pic * range.size(); i < ie; ++i)
	{
		ion->insert(p_generator(rnd_gen));
	}

	VERBOSE << save("H0", ion->dataset()) << std::endl;

	ion->rehash();

	VERBOSE << save("H1", ion->dataset()) << std::endl;

	ion->sync();

	ion->wait();

	ion->rehash();

	VERBOSE << save("H2", ion->dataset()) << std::endl;

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

