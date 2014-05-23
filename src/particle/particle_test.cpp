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
#include "../../applications/particle_solver/pic_engine_default.h"

using namespace simpla;

template<typename TParam>
class TestParticle: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(20);
		TParam::SetUpMesh(&mesh);
		cfg_str = "n0=function(x,y,z)"
				"  return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
				" end "
				"ion={ Name=\"H\",Mass=1.0e-31,Charge=1.6021892E-19 ,PIC=500,Temperature=300 ,Density=n0"
				"}";

		enable_sorting = TParam::ICASE / 100 > 0;

	}
public:
	typedef typename TParam::engine_type engine_type;

	typedef typename engine_type::Point_s Point_s;

	typedef typename engine_type::scalar_type scalar_type;

	typedef typename TParam::mesh_type mesh_type;

	typedef Particle<engine_type> particle_pool_type;

	typedef typename mesh_type::iterator iterator;

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

	LuaObject cfg;

	cfg.ParseString(TestFixture::cfg_str);

	pool_type ion(mesh,cfg["ion"]);

}
}

TYPED_TEST_P(TestParticle,scatter_n){
{

	typedef typename TestFixture::mesh_type mesh_type;

	typedef typename TestFixture::particle_pool_type pool_type;

	typedef typename TestFixture::Point_s Point_s;

	typedef typename TestFixture::iterator iterator;

	typedef typename TestFixture::coordinates_type coordinates_type;

	typedef typename TestFixture::scalar_type scalar_type;

	mesh_type const & mesh = TestFixture::mesh;

	LuaObject cfg;
	cfg.ParseString(TestFixture::cfg_str);

	Field<mesh_type,pool_type::IForm,scalar_type> n(mesh), n0(mesh);

	pool_type ion(mesh,cfg["ion"]);

	Field<mesh_type,EDGE,Real> E(mesh);
	Field<mesh_type,FACE,Real> B(mesh);

	E.Clear();
	B.Clear();
	n0.Clear();
	n.Clear();

	ion.Scatter(&n, E,B );

	GLOBAL_DATA_STREAM.OpenFile("ParticleTest");
	GLOBAL_DATA_STREAM.OpenGroup("/");
	LOGGER<<SAVE1( n );
	LOGGER<<SAVE1(ion );
	LOGGER<<Save(ion.n ,"ion_n");
	Real q=ion.q;
	{
		Real variance=0.0;

		scalar_type average=0.0;

		LuaObject n_obj=cfg["ion"]["Density"];

		Real pic =cfg["ion"]["PIC"].template as<Real>();

		for(auto s:mesh.GetRange(pool_type::IForm))
		{
			coordinates_type x=mesh.GetCoordinates(s);

			Real expect=q*n_obj(x[0],x[1],x[2]).template as<Real>();

			n0[s]=expect;

			scalar_type actual= n.get(s);

			average+=abs(actual);

			variance+=std::pow(abs (expect-actual),2.0);
		}

		if(std::is_same<typename TestFixture::engine_type,PICEngineDefault<mesh_type> >::value)
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

	LOGGER<<SAVE1(n0 );
}
}

//TYPED_TEST_P(TestParticle,move){
//{
//	GLOBAL_DATA_STREAM.OpenFile("ParticleTest");
//	GLOBAL_DATA_STREAM.OpenGroup("/");
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	typedef typename TestFixture::particle_pool_type pool_type;
//
//	typedef typename TestFixture::Point_s Point_s;
//
//	typedef typename TestFixture::iterator iterator;
//
//	typedef typename TestFixture::coordinates_type coordinates_type;
//
//	typedef typename TestFixture::scalar_type scalar_type;
//
//	mesh_type const & mesh = TestFixture::mesh;
//
//	LuaObject cfg;
//	cfg.ParseString(TestFixture::cfg_str);
//
//	Field<mesh_type,VERTEX,scalar_type> n0(mesh);
//
//	pool_type ion(mesh,cfg["ion"]);
//	ion.SetParticleSorting(TestFixture::enable_sorting);
//	Field<mesh_type,EDGE,Real> E(mesh);
//	Field<mesh_type,FACE,Real> B(mesh);
//
//	Field<mesh_type,EDGE,scalar_type> J0(mesh);
//
//	n0.Clear();
//	J0.Clear();
//	E.Clear();
//	B.Clear();
//
//	constexpr Real PI=3.141592653589793;
//
//	nTuple<3,Real> E0=
//	{	1.0e-4,1.0e-4,1.0e-4};
//	nTuple<3,Real> Bv=
//	{	0,0,1.0};
//	nTuple<3,Real> k=
//	{	2.0*PI,4.0*PI,2.0*PI};
//
//	Real q=ion.GetCharge();
//
//	auto n0_cfg= cfg["ion"]["Density"];
//
//	Real pic =cfg["ion"]["PIC"].template as<Real>();
//
//	for(auto s:mesh.GetRange(VERTEX))
//	{
//		auto x =mesh.GetCoordinates(s);
//		n0[s]=q* n0_cfg(x[0],x[1],x[2]).template as<Real>();
//	}
//
//	for (auto s : mesh.GetRange(EDGE))
//	{
//		auto x=mesh.GetCoordinates(s);
//
//		nTuple<3,Real> Ev;
//
//		Ev=E0*std::sin(Dot(k,mesh.GetCoordinates(s)));
//
//		E[s]=mesh.Sample(Int2Type<EDGE>(),s,Ev);
//	}
//
//	for (auto s : mesh.GetRange(FACE))
//	{
//		B[s]= mesh.Sample(Int2Type<FACE>(),s,Bv);
//	}
//
//	Real dt=1.0e-12;
//	Real a=0.5*(dt*q/ion.GetMass());
//
//	J0=2*n0*a*(E+a* Cross(E,B)+a*a* Dot(E,B)*B)/(1.0+Dot(Bv,Bv)*a*a);
//
//	LOG_CMD(ion.NextTimeStep(dt,E, B));
//
//	LOGGER<<SAVE1(E);
//	LOGGER<<SAVE1(B);
//	LOGGER<<SAVE1(n0 );
//	LOGGER<<SAVE1(J0 );
//	LOGGER<<SAVE1(ion.J);
//	LOGGER<<SAVE1(ion.n);
//	Real variance=0.0;
//
//	Real average=0.0;
//
//	for(auto s:mesh.GetRange(VERTEX))
//	{
//		auto expect=J0[s];
//
//		auto actual=ion.J[s];
//
//		average+=abs(expect);
//
//		variance+=std::pow(abs (expect-actual),2.0);
//	}
//
//	{
//		Real relative_error=std::sqrt(variance)/abs(average);
//
//		CHECK(relative_error);
//		EXPECT_LE(relative_error,1.0/std::sqrt(pic))<<mesh.GetDimensions();
//	}
//
//}
//}

REGISTER_TYPED_TEST_CASE_P(TestParticle, load_save, scatter_n);

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

		nTuple<3, Real> xmin = { 0, 0.0, 0 };

		nTuple<3, Real> xmax = { 1.0, 10, 2.0 };

		nTuple<3, size_t> dims[] = {

		17, 33, 65,

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

TestPICParam<Mesh, PICEngineDefault<Mesh>, 1> //,
//
//TestPICParam<Mesh, PICEngineDefault<Mesh>, 3>,
//
//TestPICParam<Mesh, PICEngineDefault<Mesh>, 3>,
//
//TestPICParam<Mesh, PICEngineDefault<Mesh>, 5>,
//
//        TestPICParam<Mesh, PICEngineDeltaF<Mesh, Real>, 1> ,
//
//TestPICParam<Mesh, PICEngineDeltaF<Mesh, Complex>, 0>,
//
//TestPICParam<Mesh, PICEngineGGauge<Mesh, Real>, 0>,
//
//TestPICParam<Mesh, PICEngineGGauge<Mesh, Real, 16>, 0>,
//
//TestPICParam<Mesh, PICEngineGGauge<Mesh, Complex>, 0>,
//
//        TestPICParam<Mesh, PICEngineDefault<Mesh>, 100>,
//
//        TestPICParam<Mesh, PICEngineDeltaF<Mesh, Real>, 100>,
//
//        TestPICParam<Mesh, PICEngineGGauge<Mesh, Real>, 100>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestParticle, ParamList);
