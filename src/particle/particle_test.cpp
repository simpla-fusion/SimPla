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
#include "../utilities/lua_state.h"
#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_euclidean.h"
#include "../parallel/parallel.h"
#include "../parallel/update_ghosts.h"

#include "particle.h"
#include "save_particle.h"

using namespace simpla;

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

struct Point_s
{
	nTuple<3, Real> x;
	nTuple<3, Real> v;
	Real f;
};
typedef ParticlePool<TMesh, Point_s> pool_type;

class TestParticle: public testing::TestWithParam<

std::tuple<

nTuple<TMesh::NDIMS, size_t>,

typename TMesh::coordinates_type,

typename TMesh::coordinates_type>

>
{
protected:
	virtual void SetUp()
	{
		GLOBAL_COMM.Init(0,nullptr);
		LOG_STREAM.SetStdOutVisableLevel(12);

		auto param = GetParam();

		mesh.SetExtents(std::get<0>(param), std::get<1>(param), std::get<2>(param));

		mesh.Decompose();

		cfg_str = "n0=function(x,y,z)"
		"  return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
		" end "
		"ion={ Name=\"H\",Mass=1.0e-31,Charge=1.6021892E-19 ,PIC=500,Temperature=300 ,Density=n0"
		"}";

	}
public:

	typedef typename pool_type::mesh_type mesh_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;

	std::string cfg_str;

	bool enable_sorting;

};
TEST_P(TestParticle,Add)
{
	pool_type p(mesh);

	auto buffer = p.GetCell();

	auto extent = mesh.GetExtents();

	rectangle_distribution<mesh_type::GetNumOfDimensions()> x_dist(extent.first, extent.second);
	std::mt19937 rnd_gen(mesh_type::GetNumOfDimensions());

	nTuple<3, Real> v =
	{ 0, 0, 0 };
	nTuple<3, Real> x =
	{ 0, 0, 0 };
	int pic = (GLOBAL_COMM.GetRank() +1)*10;

	for (auto s : mesh.GetRange(VERTEX))
	{

		for (int i = 0; i < pic; ++i)
		{
			x_dist(rnd_gen, &x[0]);

			x = mesh.CoordinatesLocalToGlobal(s, x);

			buffer.emplace_back(Point_s(
			{ x, v, 1.0 }));
		}
	}

	p.Add(&buffer);
	INFORM << "Add particle DONE " << p.size() << std::endl;

	p.Sort();
	INFORM << "Sort particle DONE " << p.size() << std::endl;

	auto dims = mesh.GetDimensions();
	dims /= 2;
	nTuple<3, size_t> start =
	{ 0, 0, 0 };

	for (auto const & v : p.data_)
	{
		CHECK(mesh.Hash(v.first)) << " " << v.second.size();

	}
//
//	for (auto const & s : mesh.GetRange(pool_type::IForm))
//	{
//		CHECK(mesh.Hash(s));
//	}

	p.Remove(p.SelectCell(start, dims));
	INFORM << "Remove particle DONE " << p.size() << std::endl;
	CHECK(p.data_.size());

	UpdateGhosts(&p);
	INFORM << "UpdateGhosts particle DONE " << p.size() << std::endl;
}

//TEST_P(TestParticle,scatter_n)
//{
//
//	LuaObject cfg;
//
//	cfg.ParseString(cfg_str);
//
//	Field<mesh_type, VERTEX, scalar_type> n(mesh), n0(mesh);
//
//	pool_type ion(mesh, cfg["ion"]);
//
//	Field<mesh_type, EDGE, Real> E(mesh);
//	Field<mesh_type, FACE, Real> B(mesh);
//
//	E.Clear();
//	B.Clear();
//	n0.Clear();
//	n.Clear();
//
//	ion.Scatter(&n, E, B);
//
//	GLOBAL_DATA_STREAM.OpenFile("ParticleTest");
//	GLOBAL_DATA_STREAM.OpenGroup("/");
//	LOGGER << SAVE(n);
//	LOGGER << SAVE(ion);
//	LOGGER << Save("ion_n", ion.n);
//	Real q = ion.q;
//	{
//		Real variance = 0.0;
//
//		scalar_type average = 0.0;
//
//		LuaObject n_obj = cfg["ion"]["Density"];
//
//		Real pic = cfg["ion"]["PIC"].template as<Real>();
//
//		for (auto s : mesh.GetRange(VERTEX))
//		{
//			coordinates_type x = mesh.GetCoordinates(s);
//
//			Real expect = q * n_obj(x[0], x[1], x[2]).template as<Real>();
//
//			n0[s] = expect;
//
//			scalar_type actual = n.get(s);
//
//			average += abs(actual);
//
//			variance += std::pow(abs(expect - actual), 2.0);
//		}
//
//		if (std::is_same<engine_type, PICEngineDefault<mesh_type> >::value)
//		{
//			Real relative_error = std::sqrt(variance) / abs(average);
//			CHECK(relative_error);
//			EXPECT_LE(relative_error, 1.0 / std::sqrt(pic));
//		}
//		else
//		{
//			Real error = 1.0 / std::sqrt(static_cast<double>(ion.size()));
//
//			EXPECT_LE(abs(average), error);
//		}
//
//	}
//
//	LOGGER << SAVE(n0);
//
//}

//TYPED_TEST_P(TestParticle,move){
//{
//	GLOBAL_DATA_STREAM.OpenFile("ParticleTest");
//	GLOBAL_DATA_STREAM.OpenGroup("/");
//	typedef mesh_type mesh_type;
//
//	typedef particle_pool_type pool_type;
//
//	typedef Point_s Point_s;
//
//	typedef iterator iterator;
//
//	typedef coordinates_type coordinates_type;
//
//	typedef scalar_type scalar_type;
//
//	mesh_type const & mesh = mesh;
//
//	LuaObject cfg;
//	cfg.ParseString(cfg_str);
//
//	Field<mesh_type,VERTEX,scalar_type> n0(mesh);
//
//	pool_type ion(mesh,cfg["ion"]);
//	ion.SetParticleSorting(enable_sorting);
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
INSTANTIATE_TEST_CASE_P(FETL, TestParticle,

testing::Combine(testing::Values(

nTuple<3, size_t>(
{ 12, 16, 10 }) //
// ,nTuple<3, size_t>( { 1, 1, 1 }) //
//        , nTuple<3, size_t>( { 17, 1, 1 }) //
//        , nTuple<3, size_t>( { 1, 17, 1 }) //
//        , nTuple<3, size_t>( { 1, 1, 10 }) //
//        , nTuple<3, size_t>( { 1, 10, 20 }) //
//        , nTuple<3, size_t>( { 17, 1, 17 }) //
//        , nTuple<3, size_t>( { 17, 17, 1 }) //
		//        , nTuple<3, size_t>( { 12, 16, 10 })

		),

testing::Values(nTuple<3, Real>(
{ 0.0, 0.0, 0.0, })  //
//        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })

		),

testing::Values(

nTuple<3, Real>(
{ 1.0, 2.0, 3.0 })  //
//        , nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
//        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
//        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
//        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
//        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
//        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //

		)));

