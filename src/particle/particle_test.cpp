/*
 * particle_test.cpp
 *
 *  Created on: 2013年11月6日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <random>

#include "../fetl/fetl.h"
#include "../fetl/save_field.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_euclidean.h"

#include "particle.h"
#include "save_particle.h"
#include "pic_engine_full.h"
#include "pic_engine_deltaf.h"
#include "pic_engine_ggauge.h"

using namespace simpla;

template<typename TParam>
class TestParticle: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);
		TParam::SetUpMesh(&mesh);
		cfg_str = "n0=function(x,y,z)"
				"  return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
				" end "
				"ion={ Name=\"H\",Mass=1.0e-31,Charge=1.6021892E-19 ,PIC=40,Temperature=1.0e-34 ,Density=n0"
				"}";

		enable_sorting = TParam::ICASE / 100 > 0;

	}
public:
	typedef typename TParam::engine_type engine_type;

	typedef typename engine_type::Point_s Point_s;

	typedef typename engine_type::scalar_type scalar_type;

	typedef typename TParam::mesh_type mesh_type;

	typedef Particle<engine_type> particle_pool_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;

	std::string cfg_str;

	bool enable_sorting;

};

TYPED_TEST_CASE_P(TestParticle);

TYPED_TEST_P(TestParticle,load_save){
{
	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	mesh_type const & mesh = TestFixture::mesh;

	pool_type ion(mesh);

	ion.Update();

	LuaObject cfg;
	cfg.ParseString(TestFixture::cfg_str);

	ion.Load(cfg["ion"]);

}
}

TYPED_TEST_P(TestParticle,scatter_n){
{

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	typedef typename TestFixture::index_type index_type;

	typedef typename TestFixture::coordinates_type coordinates_type;

	typedef typename TestFixture::scalar_type scalar_type;

	mesh_type const & mesh = TestFixture::mesh;

	pool_type ion(mesh);

	ion.Update();

	LuaObject cfg;
	cfg.ParseString(TestFixture::cfg_str);

	Field<mesh_type,VERTEX,scalar_type> n(mesh), n0(mesh);

	ion.Load(cfg["ion"]);

	Field<mesh_type,EDGE,Real> E(mesh);
	Field<mesh_type,FACE,Real> B(mesh);

	E.Fill(1.0);
	B.Fill(1.0);
	n.Fill(0);
	n0.Fill(0);

	ion.Scatter(&n,E,B);

//	GLOBAL_DATA_STREAM.OpenFile("ParticleTest");
//	GLOBAL_DATA_STREAM.OpenGroup("/");
//	LOGGER<<DUMP1(n );
//	LOGGER<<DUMP1(ion );

	{
		Real variance=0.0;

		scalar_type average=0.0;

		LuaObject n_obj=cfg["ion"]["Density"];

		Real pic =cfg["ion"]["PIC"].template as<Real>();

		for(auto s:mesh.GetRange(VERTEX))
		{
			coordinates_type x=mesh.GetCoordinates(s);

			Real expect=n_obj(x[0],x[1],x[2]).template as<Real>();

			n0[s]=expect;

			scalar_type actual= n.get(s);

			average+=actual;

			variance+=std::pow(abs (expect-actual),2.0);
		}

		if(std::is_same<typename TestFixture::engine_type,PICEngineFull<mesh_type> >::value)
		{
			Real relative_error=std::sqrt(variance)/abs(average);
			CHECK(relative_error);
			EXPECT_LE(relative_error,1.0/std::sqrt(pic));
		}
		else
		{
			Real error=1.0/std::sqrt(static_cast<double>(ion.size()));

			EXPECT_LE(abs(average),error);
		}

	}

//	LOGGER<<DUMP1(n0 );

}
}

TYPED_TEST_P(TestParticle,move){
{

	GLOBAL_DATA_STREAM.OpenFile("ParticleTest");
	GLOBAL_DATA_STREAM.OpenGroup("/");

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	typedef typename TestFixture::index_type index_type;

	typedef typename TestFixture::coordinates_type coordinates_type;

	typedef typename TestFixture::scalar_type scalar_type;

	mesh_type const & mesh = TestFixture::mesh;

	pool_type ion(mesh);

	ion.Update();

	ion.SetParticleSorting(TestFixture::enable_sorting);

	LuaObject cfg;
	cfg.ParseString(TestFixture::cfg_str);

	Field<mesh_type,VERTEX,scalar_type> n(mesh), n0(mesh);

	ion.Load(cfg["ion"]);

	Field<mesh_type,EDGE,Real> E(mesh);
	Field<mesh_type,FACE,Real> B(mesh);

	Field<mesh_type,EDGE,scalar_type> J(mesh);

	Field<mesh_type,EDGE,Real> J0(mesh);

	n.Clear();
	J.Clear();
	n0.Clear();
	J0.Clear();
	E.Clear();
	B.Clear();

	constexpr Real PI=3.141592653589793;

	Real dt = 1.0e-10;

	nTuple<3,Real> E0=
	{	1.0e-4,0.0e-4,0.0e-4};
	nTuple<3,Real> Bv=
	{	0,0,0};
	nTuple<3,Real> k=
	{	2.0*PI,4.0*PI,6.0*PI};

	auto n0_cfg= cfg["ion"]["Density"];

	Real pic =cfg["ion"]["PIC"].template as<Real>();

	for(auto s:mesh.GetRange(VERTEX))
	{
		auto x =mesh.GetCoordinates(s);
		n0[s]= n0_cfg(x[0],x[1],x[2]).template as<Real>();
	}

	for (auto s : mesh.GetRange(EDGE))
	{
		E[s]=mesh.Volume(s);
	}

	LOGGER<<Dump(E,"Volume",false);

	for (auto s : mesh.GetRange(EDGE))
	{
		auto x=mesh.GetCoordinates(s);

		nTuple<3,Real> Ev;

		Ev=E0*std::sin(Dot(k,mesh.GetCoordinates(s)));

		E[s]=mesh.Sample(Int2Type<EDGE>(),s,Ev);
	}

	for (auto s : mesh.GetRange(FACE))
	{
		B[s]=mesh.Sample(Int2Type<FACE>(),s,Bv);
	}
	J0=n0*E*(dt*ion.GetCharge()/ion.GetMass());

	LOGGER<<DUMP1(E);
	LOGGER<<DUMP1(n0 );
	LOGGER<<DUMP1(J0 );

	LOG_CMD(ion.NextTimeStep(dt,E, B));

	ion.Scatter(&J ,E,B);
	ion.Scatter(&n ,E,B);

	LOGGER<<DUMP1(J);
	LOGGER<<DUMP1(n);

	Real variance=0.0;

	scalar_type average=0.0;

	for(auto s:mesh.GetRange(EDGE))
	{
		auto expect=J0[s];

		auto actual=J[s];

		average+=abs(actual);

		variance+=std::pow(abs (expect-actual),2.0);
	}

	{
		Real relative_error=std::sqrt(variance)/abs(average);
		CHECK(relative_error);
		EXPECT_LE(relative_error,2.0/std::sqrt(pic))<<mesh.GetDimensions();
	}

}
}

REGISTER_TYPED_TEST_CASE_P(TestParticle, load_save, scatter_n, move);

template<typename TM, typename TEngine, int CASE> struct TestPICParam;
typedef RectMesh<OcForest, EuclideanGeometry> Mesh;

template<typename TEngine, int CASE>
struct TestPICParam<Mesh, TEngine, CASE>
{

	static constexpr int ICASE = CASE;
	typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;

	typedef TEngine engine_type;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin =
		{ 0, -1.0, 0 };

		nTuple<3, Real> xmax =
		{ 1.0, 10, 1.0 };

		nTuple<3, size_t> dims[] =
		{ 17, 33, 65,

		17, 1, 1,

		1, 17, 1,

		1, 1, 17,

		1, 17, 17,

		17, 1, 17,

		17, 17, 1,

		17, 17, 17

		};
		mesh->SetExtent(xmin, xmax);

		mesh->SetDimensions(dims[ICASE % 100]);

		mesh->Update();

	}

};

typedef testing::Types<

//TestPICParam<Mesh, PICEngineFull<Mesh>, 0>,
//
//TestPICParam<Mesh, PICEngineFull<Mesh>, 1>,

		TestPICParam<Mesh, PICEngineFull<Mesh>, 3> //,

//TestPICParam<Mesh, PICEngineFull<Mesh>, 5>,
//
//TestPICParam<Mesh, PICEngineFull<Mesh>, 101>,
//
//TestPICParam<Mesh, PICEngineFull<Mesh>, 103>,
//
//TestPICParam<Mesh, PICEngineFull<Mesh>, 105>,
//
//
//TestPICParam<Mesh, PICEngineDeltaF<Mesh, Real>, 0>,
//
//TestPICParam<Mesh, PICEngineFull<Mesh>, 100>,
//
//TestPICParam<Mesh, PICEngineDeltaF<Mesh, Real>, 100>
//
//TestPICParam<Mesh, PICEngineGGauge<Mesh, Real>, 0>,
//
//TestPICParam<Mesh, PICEngineDeltaF<Mesh, Complex>, 0>,
//
//TestPICParam<Mesh, PICEngineGGauge<Mesh, Complex>, 0>
//
//, PICEngineGGauge<RectMesh<>, Real, 32>
//
//, PICEngineGGauge<RectMesh<>, Complex, 8>
//
//, PICEngineGGauge<RectMesh<>, Complex, 32>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestParticle, ParamList);
