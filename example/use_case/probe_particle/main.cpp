/*
 * main.cpp
 *
 *  Created on: 2014年10月29日
 *      Author: salmon
 */

#include "../../../core/common.h"
#include "../../../core/particle/particle.h"
#include "../../../core/manifold/fetl.h"
#include "../../../core/field/domain_dummy.h"
#include "../../../core/physics/physical_constants.h"

using namespace simpla;

struct PICDemo
{
	typedef PICDemo this_type;
	typedef Vec3 coordinates_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	SP_DEFINE_POINT_STRUCT(Point_s,
			coordinates_type ,x,
			Vec3, v,
			Real, f,
			scalar_type, w)

	SP_DEFINE_PROPERTIES(
			Real, mass,
			Real, charge,
			Real, temperature
	)

private:
	Real cmr_, q_kT_;
public:

	PICDemo() :
			mass(1.0), charge(1.0), temperature(1.0)
	{
		update();
	}

	void update()
	{
		DEFINE_PHYSICAL_CONST
		cmr_ = charge / mass;
		q_kT_ = charge / (temperature * boltzmann_constant);
	}

	~PICDemo()
	{
	}

	static std::string get_type_as_string()
	{
		return "PICDemo";
	}

	template<typename TJ, typename TE, typename TB>
	void next_timestep(Point_s * p, Real dt, TE const &fE, TB const & fB) const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + cross(p->v, t);

		v_ = cross(v_, t) / (dot(t, t) + 1.0);

		p->v += v_;
		auto a = (-dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->v += v_;
		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

	}

	static inline Point_s push_forward(coordinates_type const & x,
			Vec3 const &v, scalar_type f)
	{
		return std::move(Point_s( { x, v, f }));
	}

	static inline auto pull_back(Point_s const & p)
	DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))
};

typedef Manifold<CartesianCoordinates<StructuredMesh> > TMesh;

typedef TMesh mesh_type;

int main(int argc, char **argv)
{

	mesh_type mesh;

	nTuple<Real, 3> xmin = { 0, 0, 0 };
	nTuple<Real, 3> xmax = { 20, 2, 2 };

	nTuple<size_t, 3> dims = { 20, 1, 1 };

	mesh.dimensions(dims);
	mesh.extents(xmin, xmax);

	mesh.update();

	Real charge = 1.0, mass = 1.0, Te = 1.0;

	ProbeParticle<mesh_type, PICDemo> p(mesh, mass, charge, Te);

	p.save("/H");

//	auto buffer = p.create_child();
//
//	auto extents = mesh.extents();
//
//	rectangle_distribution<mesh_type::ndims> x_dist(nTuple<Real,3>( { 0, 0, 0 }), nTuple<Real,3>( { 1, 1, 1 }));
//
//	std::mt19937 rnd_gen(mesh_type::ndims);
//
//	nTuple<Real,3> v = { 1, 2, 3 };
//
//	int pic = 500;
//
//	auto n = [](typename mesh_type::coordinates_type const & x )
//	{
//		return 2.0; //std::sin(x[0]*TWOPI);
//	    };
//
//	auto T = [](typename mesh_type::coordinates_type const & x )
//	{
//		return 1.0;
//	};
//
//	p.properties("DumpParticle", true);
//	p.properties("ScatterN", true);

//	init_particle(&p, mesh.select(VERTEX), 500, n, T);
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

