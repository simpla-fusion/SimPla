/*
 * demo_trace.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "demo_probe_particle.h"

#include <stddef.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../../../core/parallel/parallel.h"

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
	size_t step_length = 10;
	Real dt = 0.001;

	options.register_cmd_line_option<size_t>("NUMBER_OF_STEPS", "n");

	options.register_cmd_line_option<size_t>("STRIDES", "s");

	options.register_cmd_line_option<Real>("DT", "dt");

	if (options["SHOW HELP"])
	{
		SHOW_OPTIONS("-n,--number_of_steps <NUMBER_OF_STEPS>",
				"number of steps = <NUMBER_OF_STEPS> ,default="
						+ value_to_string(num_of_steps));
		SHOW_OPTIONS("-s,--strides <STRIDES>",
				" dump record per <STRIDES> steps, default="
						+ value_to_string(step_length));
		SHOW_OPTIONS("-dt  <DT>",
				" value of time step,default =" + value_to_string(dt));

		return;
	}

	options["NUMBER_OF_STEPS"].as(&num_of_steps);

	options["STRIDES"].as<size_t>(&step_length);

	options["DT"].as<Real>(&dt);

	if (options["SHOW_HELP"])
	{
		return;
	}

	auto ion = ProbeParticle<ProbeDemo>::create();

	ion->mass = 1.0;
	ion->charge = -1.0;
	ion->temperature = 1.0e-4;

	ion->update();

	MESSAGE << endl;
	MESSAGE << "======== Summary ========" << endl;
	MESSAGE << (" ion") << " = " << "{" << *ion << "}" << endl;
	MESSAGE << (" time step") << " = " << num_of_steps << endl;
	MESSAGE << (" dt") << " = " << dt << endl;
	MESSAGE << "=========================" << endl;

	if (!(options["JUST A TEST"]))
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

		ion->on_cache_full = [&](ProbeParticle<ProbeDemo>& p)
		{
			VERBOSE << save("ion", p.cache(), SP_APPEND) << endl;
		};

		ion->next_n_timesteps(num_of_steps, 0.0, dt, E, B);

	}
}

