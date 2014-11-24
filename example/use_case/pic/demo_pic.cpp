/*
 * pic_case.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include <string>
#include "../../../core/application/use_case.h"
#include "demo_pic.h"

USE_CASE(pic)
{

	bool is_configure_test_ = false;

	size_t timestep = 10;

	double dt = 0.01;

	parse_cmd_line(

	[&](std::string const & opt,std::string const & value)->int
	{

		if (opt=="t"|| opt=="test")
		{
			is_configure_test_=true;
		}
		else if(opt=="n" )
		{
			timestep=ToValue<size_t>(value);
		}
		else if(opt=="dt" )
		{
			dt=ToValue<double>(value);
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

	typedef Manifold<CartesianCoordinates<StructuredMesh> > TManifold;

	typedef TManifold manifold_type;

	auto manifold = make_manifold<TManifold>();

	manifold->load(DICT["Mesh"]);

	manifold->update();

	//	Particle<manifold_type, PICDemo, PolicyProbeParticle>

	KineticParticle<PICDemo, manifold_type> ion(manifold);

	ion.load(DICT["Particle"]);
	ion.update();

	STDOUT << std::endl;
	STDOUT << "======== Summary ========" << std::endl;
	RIGHT_COLUMN(" mesh" ) << " = {" << *manifold << "}" << std::endl;
	RIGHT_COLUMN(" ion") << " = " << "{" << ion << "}" << std::endl;
	RIGHT_COLUMN(" time step" ) << " = " << timestep << std::endl;
	RIGHT_COLUMN(" dt" ) << " = " << dt << std::endl;
	STDOUT << "=========================" << std::endl;

	if (!is_configure_test_)
	{

//	ion.push_back(p);
//
//	auto n = [](typename manifold_type::coordinates_type const & x )
//	{	return 2.0;};
//
//	auto T = [](typename manifold_type::coordinates_type const & x )
//	{	return 1.0;};
//
//	init_particle(make_domain<VERTEX>(manifold), 5, n, T, &ion);
//
//	auto B = [](nTuple<Real,3> const & )
//	{
//		return nTuple<Real,3>(
//				{	0,0,2});
//	};
//	auto E = [](nTuple<Real,3> const & )
//	{
//		return nTuple<Real,3>(
//				{	0,0,2});
//	};
//
//	ion.next_n_steps(timestep, dt, E, B);
	}
}

