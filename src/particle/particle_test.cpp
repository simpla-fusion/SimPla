/*
 * particle_test.cpp
 *
 *  created on: 2013-11-6
 *      Author: salmon
 */

//#include "particle_test.h"
//
//INSTANTIATE_TEST_CASE_P(Particle, TestParticle, testing::Combine(
//
//testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })  //
////        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })
//
//        ),
//
//testing::Values(
//
//nTuple<3, Real>( { 1.0, 2.0, 3.0 })  //
////        , nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
////        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
////        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
////        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
////        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
////        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //
//
//        ),
//
//testing::Values(
//
//nTuple<3, size_t>( { 10, 10, 1 }) //
//        // ,nTuple<3, size_t>( { 1, 1, 1 }) //
//        //        , nTuple<3, size_t>( { 17, 1, 1 }) //
//        //        , nTuple<3, size_t>( { 1, 17, 1 }) //
//        //        , nTuple<3, size_t>( { 1, 1, 10 }) //
//        //        , nTuple<3, size_t>( { 1, 10, 20 }) //
//        //        , nTuple<3, size_t>( { 17, 1, 17 }) //
//        //        , nTuple<3, size_t>( { 17, 17, 1 }) //
//        //        , nTuple<3, size_t>( { 12, 16, 10 })
//
//        )));
#include <random>
#include "../fetl/fetl.h"
#include "../fetl/save_field.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/lua_state.h"

#include "../parallel/parallel.h"
#include "../parallel/mpi_aux_functions.h"
#include "particle.h"
#include "particle_update_ghosts.h"
//
//#include "save_particle.h"
#include "particle_pool.h"
using namespace simpla;

#include "../mesh/mesh_rectangle.h"
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"
#include "../../applications/particle_solver/pic_engine_default.h"
typedef Mesh<CartesianGeometry<UniformArray>, false> TMesh;

typedef TMesh mesh_type;

int main(int argc, char **argv)
{
	LOG_STREAM.set_stdout_visable_level(12);
	GLOBAL_COMM.init();

	typedef typename PICEngineDefault<mesh_type>::Point_s Point_s;
	typedef Particle<PICEngineDefault<mesh_type> > pool_type;

	mesh_type mesh;

	nTuple<3, Real> xmin =
	{	0, 0, 0};
	nTuple<3, Real> xmax =
	{	2, 2, 2};

	nTuple<3, size_t> dims =
	{	128, 128, 1};

	mesh.set_dimensions(dims);
	mesh.set_extents(xmin, xmax);

	mesh.Update();

	pool_type p(mesh);

	auto buffer = p.create_child();

	auto extents = mesh.get_extents();

	rectangle_distribution<mesh_type::get_num_of_dimensions()> x_dist(extents.first, extents.second);

	std::mt19937 rnd_gen(mesh_type::get_num_of_dimensions());

	nTuple<3, Real> v =
	{	1, 2, 3};

	int pic = 100;

//	if (GLOBAL_COMM.get_rank() == 0)
	{
		for (auto s : mesh.Select(VERTEX))
		{
			for (int i = 0; i < pic; ++i)
			{
				buffer.emplace_back(Point_s(
								{	mesh.CoordinatesLocalToGlobal(s, x_dist(rnd_gen)), v, 1.0}));
			}
		}

		p.Add(&buffer);

		INFORM << "Add particle DONE " << p.size() << std::endl;

	}
	//	std::vector<double> a;
	//
	//	p.Remove(
	//	        mesh.Select(pool_type::IForm, std::get<0>(extents) + (std::get<1>(extents) - std::get<0>(extents)) * 0.25,
	//	                std::get<0>(extents) + (std::get<1>(extents) - std::get<0>(extents)) * 0.75));
	//
	//	INFORM << "Remove particle DONE " << p.size() << std::endl;
	//	p.Remove(mesh.Select(pool_type::IForm));
	//
	//	INFORM << "Remove particle DONE " << p.size() << std::endl;
	//	EXPECT_NE(p.size(), 0);
	//
	//	p.clear();
	//	INFORM << "Remove particle DONE " << p.size() << std::endl;
	//
	//	for (auto const & v : p.data())
	//	{
	//		if (v.second.size() > 0)
	//			CHECK((mesh.DecompactRoot(v.first)));
	//	}

	VERBOSE << "UpdateGhosts particle DONE. Local particle number =" << (p.Count()) << std::endl;

	auto total=reduce(p.Count());

	INFORM << "UpdateGhosts particle DONE. Total particle number = " << total << std::endl;

	UpdateGhosts(&p);

	VERBOSE << "UpdateGhosts particle DONE. Local particle number =" << (p.Count()) << std::endl;

	total=reduce(p.Count());

	INFORM << "UpdateGhosts particle DONE. Total particle number = " << total << std::endl;

	p.update_fields();

	p.save("ParticleTest.h5:/");

//
//	UpdateGhosts(&p);
//	VERBOSE << "UpdateGhosts particle DONE " << p.size() << std::endl;
//
//	UpdateGhosts(&p);
//	VERBOSE << "UpdateGhosts particle DONE " << p.size() << std::endl;
}
