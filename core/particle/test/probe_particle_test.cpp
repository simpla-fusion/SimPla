/*
 * probe_particle_test.cpp
 *
 *  Created on: 2014-10-29
 *      Author: salmon
 */

#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../parallel/parallel.h"

#include "../../utilities/lua_object.h"
#include "../parallel/message_comm.h"
#include "../parallel/MPIAuxFunctions.h"
#include "../sync_particle.h"
#include "load_particle.h"
#include "tracable_particle.h"

//
//#include "save_particle.h"
using namespace simpla;
#include "../field/saveField.h"
#include "../../diff_geometry/mesh.h"
#include "../../diff_geometry/domain.h"
#include "../../diff_geometry/topology/structured.h"
#include "../../diff_geometry/geometry/cartesian.h"
#include "../../applications/particle_solver/pic_engine_fullf.h"

typedef Manifold<CartesianCoordinate<RectMesh> > TManifold;

typedef TManifold mesh_type;

int main(int argc, char **argv)
{
	LOGGER.set_MESSAGE_visable_level(12);
	GLOBAL_COMM.init();

	typedef typename PICEngineFullF::Point_s Point_s;

	typedef ProbeParticle<mesh_type, PICEngineFullF> particle_type;

	mesh_type mesh;

	nTuple<Real,3> xmin = { 0, 0, 0 };
	nTuple<Real,3> xmax = { 20, 2, 2 };

	nTuple<size_t,3> dims = { 20, 1, 1 };

	mesh.dimensions(dims);
	mesh.extents(xmin, xmax);

	mesh.update();

	GLOBAL_DATA_STREAM.properties("Force Record Storage",true);

	particle_type p(mesh);

//	auto buffer = p.create_child();
//
//	auto extents = geometry.extents();
//
//	rectangle_distribution<mesh_type::ndims> x_dist(nTuple<Real,3>( { 0, 0, 0 }), nTuple<Real,3>( { 1, 1, 1 }));
//
//	std::mt19937 rnd_gen(mesh_type::ndims);
//
//	nTuple<Real,3> v = { 1, 2, 3 };
//
//	int pic = 500;
//
//	auto n = [](typename mesh_type::coordinate_tuple const & x )
//	{
//		return 2.0; //std::sin(x[0]*TWOPI);
//	    };
//
//	auto T = [](typename mesh_type::coordinate_tuple const & x )
//	{
//		return 1.0;
//	};
//
//	p.properties("DumpParticle", true);
//	p.properties("ScatterN", true);

//	init_particle(&p, geometry.select(VERTEX), 500, n, T);
//
////	{
////		auto range=geometry.select(VERTEX);
////		auto s0=*std::get<0>(range);
////		nTuple<3,Real> r=
////		{	0.5,0.5,0.5};
////
////		particle_type::Point_s a;
////		a.x = geometry.coordinates_local_to_global(s0, r);
////		a.f = 1.0;
////		p[s0].push_back(std::move(a));
////
////	}
//
	p.save("/H");
//	p.updateFields();
//
//	p.write("/H");
//
//	INFORM << "update_ghosts particle DONE. Local particle number =" << (p.Count()) << std::endl;
//
//	INFORM << "update_ghosts particle DONE. Total particle number = " << reduce(p.Count()) << std::endl;
//
//	p.updateFields();
//
//	p.write("/H/");

//	if(GLOBAL_COMM.get_rank()==0)
//	{
//		for (auto s : geometry.select(VERTEX))
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
