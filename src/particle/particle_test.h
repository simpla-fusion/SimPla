/*
 * particle_test.h
 *
 *  Created on: 2014年6月30日
 *      Author: salmon
 */

#ifndef PARTICLE_TEST_H_
#define PARTICLE_TEST_H_
#include <gtest/gtest.h>
#include <random>
#include "../fetl/fetl.h"
#include "../fetl/save_field.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/lua_state.h"

#include "../parallel/parallel.h"

#include "particle.h"
#include "particle_update_ghosts.h"
//
//#include "save_particle.h"
#include "particle_pool.h"
using namespace simpla;

#ifndef TMESH
#include "../mesh/mesh_rectangle.h"
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"
typedef Mesh<CartesianGeometry<UniformArray>, false> TMesh;
#else
typedef TMESH TMesh;
#endif

class TestParticle: public testing::TestWithParam<
        std::tuple<typename TMesh::coordinates_type, typename TMesh::coordinates_type, nTuple<TMesh::NDIMS, size_t> > >
{
protected:
	virtual void SetUp()
	{

		auto param = GetParam();

		xmin = std::get<0>(param);
		xmax = std::get<1>(param);
		dims = std::get<2>(param);
		mesh.SetExtents(xmin, xmax, dims);

		mesh.Decompose();
//
//		cfg_str = "n0=function(x,y,z)"
//				"  return (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5) "
//				" end "
//				"ion={ Name=\"H\",Mass=1.0e-31,Charge=1.6021892E-19 ,PIC=500,Temperature=300 ,Density=n0"
//				"}";

	}
public:

	typedef TMesh mesh_type;

	struct Point_s
	{
		nTuple<3, Real> x;
		nTuple<3, Real> v;
		Real f;
	};
	typedef ParticlePool<mesh_type, Point_s> pool_type;

	mesh_type mesh;

	nTuple<3, Real> xmin, xmax;

	nTuple<3, size_t> dims;

};

TEST_P(TestParticle,Add)
{

	pool_type p(mesh);

	auto buffer = p.create_child();

	auto extents = mesh.GetExtents();

	rectangle_distribution<mesh_type::GetNumOfDimensions()> x_dist(extents.first, extents.second);

	std::mt19937 rnd_gen(mesh_type::GetNumOfDimensions());

	nTuple<3, Real> v = { 0, 0, 0 };

	int pic = (GLOBAL_COMM.GetRank() +1)*10;

	for (auto s : mesh.Select(VERTEX))
	{
		for (int i = 0; i < pic; ++i)
		{
			buffer.emplace_back(Point_s( { mesh.CoordinatesLocalToGlobal(s, x_dist(rnd_gen)), v, 1.0 }));
		}
	}

	p.Add(&buffer);

	INFORM << "Add particle DONE " << p.size() << std::endl;

	EXPECT_EQ(p.size(), mesh.GetLocalMemorySize(VERTEX) * pic);

	std::vector<double> a;

	p.Remove(
	        p.Select(

	        std::get<0>(extents) + (std::get<1>(extents) - std::get<0>(extents)) * 0.25,
	                std::get<0>(extents) + (std::get<1>(extents) - std::get<0>(extents)) * 0.75

	                )

	                );

	INFORM << "Remove particle DONE " << p.size() << std::endl;
	p.Remove(p.Select());

	INFORM << "Remove particle DONE " << p.size() << std::endl;
	EXPECT_NE(p.size(), 0);

	p.clear();
	INFORM << "Remove particle DONE " << p.size() << std::endl;

//	for (auto const & v : p.data())
//	{
//		if (v.second.size() > 0)
//			CHECK((mesh.DecompactRoot(v.first)));
//	}

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
//		for (auto s : mesh.Select(VERTEX))
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
//	for(auto s:mesh.Select(VERTEX))
//	{
//		auto x =mesh.GetCoordinates(s);
//		n0[s]=q* n0_cfg(x[0],x[1],x[2]).template as<Real>();
//	}
//
//	for (auto s : mesh.Select(EDGE))
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
//	for (auto s : mesh.Select(FACE))
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
//	for(auto s:mesh.Select(VERTEX))
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

#endif /* PARTICLE_TEST_H_ */
