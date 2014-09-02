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
//        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })
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
#include "../parallel/message_comm.h"
#include "../parallel/mpi_aux_functions.h"

#include "kinetic_particle.h"
#include "particle_update_ghosts.h"
#include "load_particle.h"

//
//#include "save_particle.h"
using namespace simpla;

#include "../mesh/mesh_rectangle.h"
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"
#include "../../applications/particle_solver/pic_engine_fullf.h"

typedef Mesh<CartesianGeometry<UniformArray>, false> TMesh;

typedef TMesh mesh_type;

int main(int argc, char **argv)
{
	LOGGER.set_stdout_visable_level(12);
	GLOBAL_COMM.init();

	typedef typename PICEngineFullF::Point_s Point_s;

	typedef KineticParticle<mesh_type, PICEngineFullF> particle_type;

	mesh_type mesh;

	nTuple<3, Real> xmin = { 0, 0, 0 };
	nTuple<3, Real> xmax = { 20, 2, 2 };

	nTuple<3, size_t> dims = { 20, 1, 1 };

	mesh.set_dimensions(dims);
	mesh.set_extents(xmin, xmax);

	mesh.update();

	GLOBAL_DATA_STREAM.set_property("Force Record Storage",true);

	particle_type p(mesh);

	auto buffer = p.create_child();

	auto extents = mesh.get_extents();

	rectangle_distribution<mesh_type::get_num_of_dimensions()> x_dist(nTuple<3, Real>( { 0, 0, 0 }), nTuple<3, Real>( {
	        1, 1, 1 }));

	std::mt19937 rnd_gen(mesh_type::get_num_of_dimensions());

	nTuple<3, Real> v = { 1, 2, 3 };

	int pic = 500;

	auto n = [](typename mesh_type::coordinates_type const & x )
	{
		return 2.0; //std::sin(x[0]*TWOPI);
	    };

	auto T = [](typename mesh_type::coordinates_type const & x )
	{
		return 1.0;
	};

	p.properties.set("DumpParticle", true);
	p.properties.set("ScatterN", true);

//	InitParticle(&p, mesh.select(VERTEX), 500, n, T);
//
////	{
////		auto range=mesh.select(VERTEX);
////		auto s0=*std::get<0>(range);
////		nTuple<3,Real> r=
////		{	0.5,0.5,0.5};
////
////		particle_type::Point_s a;
////		a.x = mesh.coordinates_local_to_global(s0, r);
////		a.f = 1.0;
////		p[s0].push_back(std::move(a));
////
////	}
//
	p.save("/H");
//	p.update_fields();
//
//	p.save("/H");
//
//	INFORM << "update_ghosts particle DONE. Local particle number =" << (p.Count()) << std::endl;
//
//	INFORM << "update_ghosts particle DONE. Total particle number = " << reduce(p.Count()) << std::endl;
//
//	p.update_fields();
//
//	p.save("/H/");

//	if(GLOBAL_COMM.get_rank()==0)
//	{
//		for (auto s : mesh.select(VERTEX))
//		{
//			rho[s]+=10;
//		}
//	}

//
//	update_ghosts(&p);
//	VERBOSE << "update_ghosts particle DONE " << p.size() << std::endl;
//
//	update_ghosts(&p);
//	VERBOSE << "update_ghosts particle DONE " << p.size() << std::endl;
}
