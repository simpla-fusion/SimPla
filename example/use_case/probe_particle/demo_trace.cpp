/*
 * demo_trace.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "demo_trace.h"

#include <string>
#include "../../../core/application/use_case.h"
#include "../../../core/particle/tracable_particle.h"

USE_CASE(pic)
{

	bool is_configure_test_ = false;
	size_t num_of_steps = 1000;
	size_t strides = 1000;
	double dt = 0.01;

	parse_cmd_line(

	[&](std::string const & opt,std::string const & value)->int
	{

		if(opt=="n" )
		{
			num_of_steps=ToValue<size_t>(value);
		}
		else if(opt=="dt" )
		{
			dt=ToValue<double>(value);
		}
		else if(opt=="s" ||opt=="strides" )
		{
			strides=ToValue<size_t>(value);
		}
		else if (opt=="t"|| opt=="test")
		{
			is_configure_test_=true;
		}
		else if(opt=="h" || opt=="help")
		{
			SHOW_OPTIONS("-n <NUM>","number of steps");
			SHOW_OPTIONS("-s <NUM>","recorder per <NUM> steps");
			SHOW_OPTIONS("-t,--test ","only read and parse input file");
			is_configure_test_=true;
			return TERMINATE;
		}
		return CONTINUE;
	}

	);

	typedef Manifold<CartesianCoordinates<StructuredMesh> > manifold_type;

	auto manifold = make_manifold<manifold_type>();

	manifold->load(DICT["Mesh"]);

	ParticleTrajectory<PICDemo, manifold_type> ion(manifold);

	ion.load(DICT["Particle"]);

	STDOUT << std::endl;
	STDOUT << "======== Summary ========" << std::endl;
	RIGHT_COLUMN(" mesh" ) << " = {" << *manifold << "}" << std::endl;
	RIGHT_COLUMN(" ion") << " = " << "{" << ion << "}" << std::endl;
	RIGHT_COLUMN(" time step" ) << " = " << num_of_steps << std::endl;
	RIGHT_COLUMN(" dt" ) << " = " << dt << std::endl;
	STDOUT << "=========================" << std::endl;

	if (is_configure_test_)
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

