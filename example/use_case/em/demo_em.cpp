/*
 * @file demo_em.cpp
 *
 *  Created on: 2014年11月28日
 *      Author: salmon
 */

#include "../../../core/application/use_case.h"
#include <memory>
#include <string>
#include "../../../core/mesh/structured/geometry.h"
#include "../../../core/mesh/structured/manifold.h"
#include "../../../core/mesh/structured/diff_scheme/fdm.h"
#include "../../../core/mesh/structured/interpolator/interpolator.h"
#include "../../../core/field/field_shared_ptr.h"
#include "../../../core/io/io.h"

using namespace simpla;

USE_CASE(em)
{
	typedef CartesianMesh mesh_type;
	size_t num_of_steps = 1000;
	size_t strides = 10;
	Real dt = 0.001;

	options.register_cmd_line_option<size_t>("NUMBER_OF_STEPS", "n");

	options.register_cmd_line_option<size_t>("STRIDES", "s");

	options.register_cmd_line_option<Real>("DT", "dt");

//	if (options["SHOW_HELP"])
//	{
//		SHOW_OPTIONS("-n,--number_of_steps <NUMBER_OF_STEPS>",
//				"number of steps = <NUMBER_OF_STEPS> ,default="
//						+ value_to_string(num_of_steps));
//		SHOW_OPTIONS("-s,--strides <STRIDES>",
//				" dump record per <STRIDES> steps, default="
//						+ value_to_string(strides));
//		return;
//	}

	options["NUMBER_OF_STEPS"].as(&num_of_steps);

	options["STRIDES"].as<size_t>(&strides);
//

	size_t dims[3] = { 10, 10, 10 };
	size_t xmin[3] = { 0, 0, 0 };
	size_t xmax[3] = { 1, 1, 1 };

	mesh_type mesh;

	mesh.dimensions(dims);
	mesh.extents(xmin, xmax);
	mesh.update();

//	mesh.load(options["Mesh"]);

	LOGGER << "[ Configuration ]" << std::endl

	<< " Description=\"" << options["Description"].as<std::string>("") << "\""
			<< std::endl

			<< " Mesh =" << std::endl << "  {" << mesh << "} " << std::endl

			<< " TIME_STEPS = " << num_of_steps << std::endl

			;
//
//	// Load initialize value
//
	auto J = make_form<EDGE, Vec3>(mesh);
	auto E = make_form<EDGE, Real>(mesh);
	auto B = make_form<FACE, Real>(mesh);
//	VERBOSE_CMD(load(options["InitValue"]["B"], &B));
//	VERBOSE_CMD(load(options["InitValue"]["E"], &E));
//	VERBOSE_CMD(load(options["InitValue"]["J"], &J));
//
//	auto E_src = make_constraint<EDGE, Real>(manifold,
//			options["Constraint"]["E"]);
//	auto J_src = make_constraint<EDGE, Real>(manifold,
//			options["Constraint"]["J"]);
//	auto B_src = make_constraint<FACE, Real>(manifold,
//			options["Constraint"]["B"]);
//
//	LOGGER << "----------  Dump input ---------- " << std::endl;
//
//	cd("/Input/");
//
//	VERBOSE << SAVE(E) << std::endl;
//	VERBOSE << SAVE(B) << std::endl;
//	VERBOSE << SAVE(J) << std::endl;
//
//	LOGGER << "----------  START ---------- " << std::endl;
//
//	cd("/Save/");
//
//	if (options["JUST_A_TEST"])
//	{
//		LOGGER << " Just test configuration!" << std::endl;
//	}
//	else
//	{
//		for (size_t s = 0; s < num_of_steps; s += strides)
//		{
//
////			E_src(&E);
////			J_src(&J);
////			B_src(&B);
//
//			E += curl(B) * dt - J;
//			B += -curl(E) * dt;
//		}
//
////		VERBOSE << SAVE(E);
////		VERBOSE << SAVE(B);
//
//	}
//
//	cd("/Output/");
//	VERBOSE << SAVE(E) << std::endl;
//	VERBOSE << SAVE(B) << std::endl;
//	VERBOSE << SAVE(J) << std::endl;
//
//	LOGGER << "----------  DONE ---------- " << std::endl;

}

