/*
 * particle_test.cpp
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#include <random>
#include <fetl/fetl.h>
#include <fetl/ntuple.h>
#include <fetl/primitives.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-internal.h>
#include <mesh/uniform_rect.h>
#include <numeric/multi_normal_distribution.h>
#include <numeric/rectangle_distribution.h>
//#include <numeric/sobol_engine.h>
#include <particle/particle.h>
#include <particle/pic_engine_default.h>
#include <utilities/log.h>

using namespace simpla;

TEST(PARTICLE_TEST,Create)
{
	DEFINE_FIELDS(UniformRectMesh)

	Mesh mesh;

	Log::Verbose(10);

	mesh.dt_ = 1.0;
	mesh.xmin_[0] = 0;
	mesh.xmin_[1] = 0;
	mesh.xmin_[2] = 0;
	mesh.xmax_[0] = 1.0;
	mesh.xmax_[1] = 1.0;
	mesh.xmax_[2] = 1.0;
	mesh.dims_[0] = 5;
	mesh.dims_[1] = 5;
	mesh.dims_[2] = 5;
	mesh.gw_[0] = 1;
	mesh.gw_[1] = 1;
	mesh.gw_[2] = 1;

	mesh.Update();

	typedef Form<0, Real> RScalarField;

	Particle<PICEngineDefault<Mesh>> p_ion(mesh, 1.0, 1.0);

	p_ion.Init(100);

	Real T = 1.0;

	std::mt19937 rnd_gen(1);

	rectangle_distribution<Mesh::NUM_OF_DIMS> x_dist;
	multi_normal_distribution<Mesh::NUM_OF_DIMS> v_dist(1.0);

	mesh.ForAll(0,

	[&](typename Mesh::index_type const & s)
	{

		x_dist.Reset(mesh.GetCellShape(s));

		for(auto & p : p_ion[s])
		{
			x_dist(rnd_gen,p.x);
			v_dist(rnd_gen,p.v);
			p.f=1.0;
		}
	}

	);

	Form<0, Real> n(mesh);
	Form<1, Real> J(mesh);
	Form<1, Real> E(mesh);
	Form<2, Real> B(mesh);

	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for (auto & v : E)
	{
		v = uniform_dist(rnd_gen);
	}

	for (auto & v : B)
	{
		v = uniform_dist(rnd_gen);
	}

	CHECK(p_ion.size());

	p_ion.Sort();
	p_ion.Push(E, B);
	p_ion.Sort();
	p_ion.ScatterN(n, E, B);
	p_ion.ScatterJ(J, E, B);

}
