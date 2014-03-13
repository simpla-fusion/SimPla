/*
 * particle_test.cpp
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <random>

#include "../fetl/fetl.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

#include "../mesh/rect_mesh.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_euclidean.h"

#include "particle.h"
#include "pic_engine_full.h"
//#include "pic_engine_deltaf.h"
//#include "pic_engine_ggauge.h"

using namespace simpla;

template<typename TEngine>
class TestParticle: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 20, 0, 0 };
		mesh.SetDimensions(dims);
		mesh.SetDt(1.0);

		mesh.Update();

		cfg.ParseString(

		"n0=function(x,y,z)"
				" return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
				" end "
				"ion={ Name=\"H\",Engine=\"Full\",m=1.0,Z=1.0,PIC=200,T=1.0e4 ,"
				"  n=n0"
				"}"

		);

	}
public:

	typedef TEngine engine_type;

	typedef Particle<TEngine> particle_pool_type;

	typedef typename TEngine::mesh_type mesh_type;

	typedef typename TEngine::Point_s Point_s;

	DEFINE_FIELDS(mesh_type)

	mesh_type mesh;

	LuaObject cfg;

};

typedef testing::Types<

PICEngineFull<RectMesh<>>

//, PICEngineFull<CoRectMesh<Complex>>
//
//, PICEngineDeltaF<CoRectMesh<Real>>
//
//, PICEngineDeltaF<CoRectMesh<Complex>>
//
//, PICEngineGGauge<CoRectMesh<Real>, 8>
//
//, PICEngineGGauge<CoRectMesh<Real>, 32>
//
//, PICEngineGGauge<CoRectMesh<Complex>, 8>
//
//, PICEngineGGauge<CoRectMesh<Complex>, 32>

> AllEngineTypes;

TYPED_TEST_CASE(TestParticle, AllEngineTypes);

TYPED_TEST(TestParticle,load_save){
{
	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	mesh_type const & mesh = TestFixture::mesh;

	pool_type ion(mesh);

	ion.Update();

	ion.Load(TestFixture::cfg["ion"]);
}
}

TYPED_TEST(TestParticle,scatter_n){
{

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	typedef typename TestFixture::index_type index_type;

	typedef typename TestFixture::coordinates_type coordinates_type;

	mesh_type const & mesh = TestFixture::mesh;

	pool_type ion(mesh);

	ion.Update();

	ion.Load(TestFixture::cfg["ion"]);

	typename TestFixture::template Form<VERTEX> n(mesh);
	typename TestFixture::template Form<EDGE> E(mesh);
	typename TestFixture::template Form<FACE> B(mesh);

	E.Fill(1.0);
	B.Fill(1.0);
	n.Fill(0);

	ion.Scatter(&n,E,B);


	{
		Real variance=0.0;

		Real average=0.0;

		auto n_obj=TestFixture::cfg["ion"]["n0"];

		Real pic =TestFixture::cfg["ion"]["PIC"].template as<Real>();

		mesh. template Traversal<VERTEX>(
				[&](index_type s)
				{
					coordinates_type x=mesh.GetCoordinates(s);

					Real expect=n_obj(x[0],x[1],x[2]).template as<Real>();

					Real actual= n.get(s);

					average+=actual;

					variance+=std::pow(abs(expect-actual),2.0);
				}
		);

		if(std::is_same<typename TestFixture::engine_type,PICEngineFull<mesh_type> >::value)
		{
			Real relative_error=std::sqrt(variance)/abs(average);

			EXPECT_LE(relative_error,1.1/std::sqrt(pic));
		}
		else
		{
			Real error=1.0/std::sqrt(static_cast<double>(ion.size()));

			EXPECT_LE(abs(average),error);
		}

	}

}
}

TYPED_TEST(TestParticle,scatter_J){
{

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	mesh_type const & mesh = TestFixture::mesh;

	typename TestFixture::template Form<EDGE> J(mesh);
	typename TestFixture::template Form<EDGE> E(mesh);
	typename TestFixture::template Form<FACE> B(mesh);

	E.Init();
	B.Init();
	J.Init();

	pool_type ion(mesh);

	ion.Update();

	ion.Load(TestFixture::cfg["ion"]);

	ion.Sort();

	ion.Scatter(&J,E,B);

}
}
//
//TYPED_TEST(TestParticle,move){
//{
//
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	typedef typename TestFixture::particle_pool_type pool_type;
//
//	typedef typename TestFixture::Point_s Point_s;
//
//	typedef typename TestFixture::coordinates_type coordinates_type;
//
//	mesh_type const & mesh = TestFixture::mesh;
//
//	TestFixture::cfg.ParseString("ion.n=1.0");
//
//	pool_type ion(mesh);
//
//	ion.Update();
//
//	ion.Load(TestFixture::cfg["ion"]);
//
//	ion.Sort();
//
//	typename TestFixture::template Form<VERTEX> n(mesh);
//	typename TestFixture::template Form<EDGE> E(mesh);
//	typename TestFixture::template Form<FACE> B(mesh);
//	typename TestFixture::template Form<EDGE> J(mesh);
//
//	E.Fill(1.0);
//	B.Fill(1.0);
//	J.Update();
//
//	std::mt19937 rnd_gen(2);
//
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	for (auto & v : E)
//	{
//		v = uniform_dist(rnd_gen);
//	}
//
//	for (auto & v : B)
//	{
//		v = uniform_dist(rnd_gen);
//	}
//
//	Real expect_n = 1.0;
//	Real error=1.0/std::sqrt(static_cast<double>(ion.size()));
//
//	if(!std::is_same<typename TestFixture::engine_type,PICEngineFull<mesh_type> >::value)
//	{
//		expect_n=0.0;
//	}
//
//	{
//		typename TestFixture::template Form<VERTEX> n(mesh);
//		n.Fill(0.0);
//		ion.Scatter(&n,E,B);
//
//		Real average_n=0.0;
//		for(auto &v :n)
//		{
//			average_n+=v;
//		}
//
//		average_n/=static_cast<Real>(mesh.GetNumOfElements(0));
//
//		EXPECT_LE(abs(expect_n-abs(average_n)),error);
//	}
//
//	LOG_CMD(ion.NextTimeStep(1.0,E, B));
//
//	{
//		typename TestFixture::template Form<VERTEX> n(mesh);
//		n.Fill(0.0);
//		ion.Scatter(&n,E,B);
//
//		Real average_n=0.0;
//
//		for(auto &v :n)
//		{
//			average_n+=v;
//		}
//		average_n/=static_cast<Real>(mesh.GetNumOfElements(0));
//
//		EXPECT_LE(abs(expect_n-average_n),error);
//
//	}
//
//}
//}
