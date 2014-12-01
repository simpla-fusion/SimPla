/*
 * demo_em.cpp
 *
 *  Created on: 2014年11月28日
 *      Author: salmon
 */

#include <stddef.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../../../core/application/use_case.h"
#include "../../../core/utilities/utilities.h"
#include "../../../core/manifold/fetl.h"

#include "../../../core/io/io.h"

using namespace simpla;

USE_CASE(em)
{

	size_t num_of_steps = 1000;
	size_t strides = 10;
	Real dt = 0.001;

	options.register_cmd_line_option<size_t>("NUMBER_OF_STEPS", "n");

	options.register_cmd_line_option<size_t>("STRIDES", "s");

	options.register_cmd_line_option<Real>("DT", "dt");

	if (options["SHOW HELP"])
	{
		SHOW_OPTIONS("-n,--number_of_steps <NUMBER_OF_STEPS>",
				"number of steps = <NUMBER_OF_STEPS> ,default="
						+ ToString(num_of_steps));
		SHOW_OPTIONS("-s,--strides <STRIDES>",
				" dump record per <STRIDES> steps, default="
						+ ToString(strides));
		SHOW_OPTIONS("-dt  <DT>",
				" value of time step,default =" + ToString(dt));

		return;
	}

	options["NUMBER_OF_STEPS"].as(&num_of_steps);

	options["STRIDES"].as<size_t>(&strides);

	options["DT"].as<Real>(&dt);

	auto manifold = Manifold<CartesianCoordinates<StructuredMesh> >::create();

	manifold->load(options["Mesh"]);

	manifold->update();

	auto J = make_form<Real, EDGE>(manifold);
	auto E = make_form<Real, EDGE>(manifold);
	auto B = make_form<Real, FACE>(manifold);

	dt = manifold->dt();

	STDOUT << std::endl;
	STDOUT << "======== Summary ========" << std::endl;
	RIGHT_COLUMN(" mesh" ) << " = {" << *manifold << "}" << std::endl;
	RIGHT_COLUMN(" time step" ) << " = " << num_of_steps << std::endl;
	RIGHT_COLUMN(" dt" ) << " = " << dt << std::endl;
	STDOUT << "=========================" << std::endl;

	if (!options["JUST A TEST"])
	{
		for (size_t s = 0; s < num_of_steps; s += strides)
		{
			E += curl(B) * dt - J;
			B += -curl(E) * dt;
		}

	}

}

