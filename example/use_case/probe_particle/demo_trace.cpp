/*
 * demo_trace.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "demo_trace.h"

#include <stddef.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../../../core/application/use_case.h"

#include "../../../core/utilities/utilities.h"
#include "../../../core/manifold/mainfold.h"
#include "../../../core/particle/particle.h"

USE_CASE(pic)
{

	size_t num_of_steps = 1000;
	size_t strides = 10;
	Real dt = 0.001;

	options.convert_cmdline_to_option<size_t>("numb_of_steps", "n",
			"numb_of_steps");

	options.convert_cmdline_to_option<size_t>("strides", "s");

	options.convert_cmdline_to_option<Real>("dt", "dt");

	if (options["SHOW_HELP"])
	{
		SHOW_OPTIONS("-n,--number_of_steps <NUM>",
				"number of steps = <NUM> ,default=" + ToString(num_of_steps));
		SHOW_OPTIONS("-s,--strides <NUM>",
				" dump record per <NUM> steps, default=" + ToString(strides));
		SHOW_OPTIONS("-dt  <real number>",
				" value of time step,default =" + ToString(dt));

		return;
	}

	options["num_of_steps"].as(&num_of_steps);

	options["strides"].as<size_t>(&strides);

	options["dt"].as<Real>(&dt);

	typedef Manifold<CartesianCoordinates<StructuredMesh> > manifold_type;

	auto manifold = make_manifold<manifold_type>();

	Particle<manifold_type, PICDemo> ion(manifold);

	manifold->load(options["Mesh"]);
	ion.load(options["Particle"]);

	STDOUT << std::endl;
	STDOUT << "======== Summary ========" << std::endl;
	RIGHT_COLUMN(" mesh" ) << " = {" << *manifold << "}" << std::endl;
	RIGHT_COLUMN(" ion") << " = " << "{" << ion << "}" << std::endl;
	RIGHT_COLUMN(" time step" ) << " = " << num_of_steps << std::endl;
	RIGHT_COLUMN(" dt" ) << " = " << dt << std::endl;
	STDOUT << "=========================" << std::endl;

	if (options["JUST_A_TEST"])
	{
		exit(0);
	}

	ion.cache_length(strides);

	manifold->update();

	ion.update();

	auto B = [](nTuple<Real,3> const & )
	{
		return nTuple<Real,3>(
				{	0,0,2});
	};
	auto E = [](nTuple<Real,3> const & )
	{
		return nTuple<Real,3>(
				{	0,0,2});
	};

	for (size_t s = 0; s < num_of_steps; s += strides)
	{
		ion.next_n_steps(num_of_steps, dt, E, B);

		save("ion", ion.dataset());

	}

}

