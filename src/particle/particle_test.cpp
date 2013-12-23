/*
 * particle_test.cpp
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#include <random>
#include <gtest/gtest.h>

#include "particle.h"
#include "pic_engine_full.h"
#include "../io/data_stream.h"
#include "save_particle.h"
#include "load_particle.h"

#include "../fetl/fetl.h"
#include "../utilities/log.h"
#include "../mesh/co_rect_mesh.h"
#include "../utilities/pretty_stream.h"

#include "../numeric/multi_normal_distribution.h"
#include "../numeric/rectangle_distribution.h"

using namespace simpla;

DEFINE_FIELDS(CoRectMesh<Real>)

template<typename TEngine>
class TestParticle: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		mesh.dt_ = 1.0;
		mesh.xmin_[0] = 0;
		mesh.xmin_[1] = 0;
		mesh.xmin_[2] = 0;
		mesh.xmax_[0] = 1.0;
		mesh.xmax_[1] = 1.0;
		mesh.xmax_[2] = 1.0;
		mesh.dims_[0] = 5;
		mesh.dims_[1] = 11;
		mesh.dims_[2] = 11;

		mesh.Update();

		GLOBAL_DATA_STREAM.OpenFile("");

	}
public:
	typedef TEngine engine_type;

	typedef Particle<TEngine> particle_pool_type;

	typedef typename TEngine::mesh_type mesh_type;

	typedef typename TEngine::Point_s Point_s;

	mesh_type mesh;

	static const size_t pic = 100;

};

typedef testing::Types<

PICEngineFull<Mesh>
//,
//
//PICEngineDeltaF<CoRectMesh<Complex>>,
//
//PICEngineDefault<CoRectMesh<Real>>,
//
//PICEngineDefault<CoRectMesh<Complex>>

> AllEngineTypes;

TYPED_TEST_CASE(TestParticle, AllEngineTypes);

TYPED_TEST(TestParticle,Create){
{

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	mesh_type const & mesh = TestFixture::mesh;

	pool_type ion(mesh);

	ion.SetMass(1.0);

	ion.SetCharge(1.0);

	ion.SetName("H");

	ion.Update();

	Real T = 1.0;

	std::mt19937 rnd_gen(1);

	rectangle_distribution<mesh_type::NUM_OF_DIMS> x_dist;

	multi_normal_distribution<mesh_type::NUM_OF_DIMS> v_dist(1.0);

	mesh.ParallelTraversal(pool_type::IForm,

			[&](typename mesh_type::index_type const & s)
			{

				typename mesh_type::coordinates_type xrange[mesh.GetCellShape(s)];

				mesh.GetCellShape(s,xrange);

				x_dist.Reset(xrange);

				nTuple<3,Real> x,v;

				for(int i=0;i<TestFixture::pic;++i)
				{
					x_dist(rnd_gen,x);
					v_dist(rnd_gen,v);
					ion.Insert(s,x,v,1.0);
				}
			}

	);

	Form<0> n(mesh);
	Form<1> J(mesh);
	Form<1> E(mesh);
	Form<2> B(mesh);

	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for (auto & v : E)
	{
		v = uniform_dist(rnd_gen);
	}

	for (auto & v : B)
	{
		v = uniform_dist(rnd_gen);
	}

	CHECK(ion.size());

	LOGGER << Data(ion,"ion");

	ion.Sort();

	ion.NextTimeStep(1.0,E, B);

	ion.Collect(&n,E,B);

	ion.Collect(&J,E,B);

}
}
