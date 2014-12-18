/*
 * demo_trace.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "demo_probe_particle.h"

#include <stddef.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../../../core/application/use_case.h"
#include "../../../core/utilities/utilities.h"
#include "../../../core/particle/particle.h"
#include "../../../core/particle/probe_particle.h"
#include "../../../core/particle/particle_engine.h"
#include "../../../core/io/io.h"
using namespace simpla;

USE_CASE(demo_probe_particle)
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

	auto ion = make_probe_particle<ProbeDemo>();

	ion->load(options["Particle"]);

	ion->update();

	STDOUT << std::endl;
	STDOUT << "======== Summary ========" << std::endl;
	RIGHT_COLUMN(" ion") << " = " << "{" << *ion << "}" << std::endl;
	RIGHT_COLUMN(" time step" ) << " = " << num_of_steps << std::endl;
	RIGHT_COLUMN(" dt" ) << " = " << dt << std::endl;
	STDOUT << "=========================" << std::endl;

	if (!options["JUST A TEST"])
	{

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

//		for (size_t s = 0; s < num_of_steps; s += strides)
//		{
//			ion->next_timestep(dt, E, B);
//		}
//		VERBOSE << save("ion", ion->dataset()) << std::endl;
	}
}

