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
#include "../fetl/save_field.h"
#include "../fetl/load_field.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

#include "../mesh/co_rect_mesh.h"

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
		mesh.dims_[0] = 200;
		mesh.dims_[1] = 1;
		mesh.dims_[2] = 1;
		mesh.dt_ = 1.0;

		mesh.Update();

		GLOBAL_DATA_STREAM.OpenFile("");

	}
public:
	typedef TEngine engine_type;

	typedef Particle<TEngine> particle_pool_type;

	typedef typename TEngine::mesh_type mesh_type;

	typedef typename TEngine::Point_s Point_s;

	mesh_type mesh;

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

	LuaObject cfg;

	cfg.ParseString(

			"n0=function(x,y,z)"
			"     return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
			" end "
			"ion={ Name=\"H\",Engine=\"Full\",m=1.0,Z=1.0,PIC=200,T=1.0e4 ,"
			"  n=n0"
			"}"

	);

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	mesh_type const & mesh = TestFixture::mesh;

	Form<0> n(mesh);
	Form<0> n0(mesh);
	Form<1> J(mesh);
	Form<1> E(mesh);
	Form<2> B(mesh);

	E.Update();
	B.Update();
	n.Update();
	J.Update();

	LoadField(cfg["n0"],&n0);

	LOGGER<<DUMP(n0);

	pool_type ion(mesh);

	ion.Update();

	ion.Deserialize(cfg["ion"]);

	std::mt19937 rnd_gen(2);

	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for (auto & v : E)
	{
		v = uniform_dist(rnd_gen);
	}

	for (auto & v : B)
	{
		v = uniform_dist(rnd_gen);
	}

	LOGGER << Data(ion,"ion");

	ion.Sort();

	LOGGER << Data(ion,"ion2");

	n.Fill(0);
	ion.Collect(&n,E,B);
	LOGGER<< " Collect "<<DUMP(n)<<DONE;

	ion.NextTimeStep(1.0,E, B);

	LOGGER<< " NextTimeStep"<<DONE;

	LOGGER << Data(ion,"ion3");

	n.Fill(0);
	ion.Collect(&n,E,B);
	LOGGER<< " Collect "<<DUMP(n)<<DONE;
//
//	ion.Collect(&J,E,B);
//	LOGGER<< " Collect "<<DUMP(J)<<DONE;

}
}
