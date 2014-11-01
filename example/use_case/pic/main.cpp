/*
 * main.cpp
 *
 *  Created on: 2014年10月29日
 *      Author: salmon
 */

#include "../../../core/common.h"
#include "../../../core/particle/particle.h"
#include "../../../core/manifold/fetl.h"
#include "../../../core/physics/physical_constants.h"
#include "../../../core/io/data_stream.h"
#include "../../../core/manifold/domain_dummy.h"
#include "../../../core/utilities/log.h"
#include "../../../core/utilities/ntuple.h"
#include "../../../core/utilities/parse_command_line.h"
#include "../../../core/parallel/message_comm.h"

using namespace simpla;

struct PICDemo2
{
	typedef PICDemo2 this_type;
	typedef Vec3 coordinates_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;

	SP_DEFINE_POINT_STRUCT(Point_s,
			coordinates_type ,x,
			Vec3, v,
			Real, f)

	SP_DEFINE_PROPERTIES(
			Real, mass,
			Real, charge,
			Real, temperature
	)
#define PROPERTIES Real  mass,	Real  charge, Real   temperature
private:
	Real cmr_, q_kT_;
public:

	PICDemo2() :
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

	~PICDemo2()
	{
	}

	static std::string get_type_as_string()
	{
		return "PICDemo2";
	}

	template<typename TJ, typename TE, typename TB>
	void next_timestep(Point_s * p, Real dt, TJ * J, TE const &fE,
			TB const & fB) const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

		Vec3 v_;

		auto t = B * (charge / mass * dt * 0.5);

		p->v += E * (charge / mass * dt * 0.5);

		v_ = p->v + cross(p->v, t);

		p->v += cross(v_, t) / (dot(t, t) + 1.0) * 2.0;

		p->v += E * (charge / mass * dt * 0.5);

		p->x += p->v * dt * 0.5;

//		scatter(p->x, p->v, p->f * charge,J);

	}

	static inline Point_s push_forward(coordinates_type const & x,
			Vec3 const &v, scalar_type f)
	{
		return std::move(Point_s(
		{ x, v, f }));
	}

	static inline auto pull_back(Point_s const & p)
	DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))
};

typedef Manifold<CartesianCoordinates<StructuredMesh> > TManifold;

typedef TManifold manifold_type;

int main(int argc, char **argv)
{
	LOGGER.set_stdout_visable_level(LOG_INFORM);
	LOGGER.init(argc, argv);
	GLOBAL_COMM.init(argc,argv);
	GLOBAL_DATA_STREAM.init(argc,argv);
	GLOBAL_DATA_STREAM.cd("/");

	size_t timestep = 10;

	double dt = 0.01;

	ParseCmdLine(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if(opt=="s" )
		{
			timestep=ToValue<size_t>(value);
		}
		else if(opt=="t" )
		{
			dt=ToValue<double>(value);
		}
		return CONTINUE;
	}

	);

	manifold_type manifold;

	nTuple<Real, 3> xmin =
	{ 0, 0, 0 };
	nTuple<Real, 3> xmax =
	{ 20, 2, 2 };

	nTuple<size_t, 3> dims =
	{ 20, 1, 1 };

	manifold.dimensions(dims);
	manifold.extents(xmin, xmax);

	manifold.update();

	Real charge = 1.0, mass = 1.0, Te = 1.0;

	Domain<manifold_type, VERTEX> domain(manifold);

	Particle<PICDemo2, Domain<manifold_type, VERTEX>> ion(domain, mass, charge,
			Te);

	INFORM << "=========================" << std::endl;
	INFORM << "dt =  \t" << dt << std::endl;
	INFORM << "time step =  \t" << timestep << std::endl;
	INFORM << "ion =  \t {" << ion << "}" << std::endl;
	INFORM << "=========================" << std::endl;

	auto n = [](typename manifold_type::coordinates_type const & x )
	{	return 2.0;};

	auto T = [](typename manifold_type::coordinates_type const & x )
	{	return 1.0;};

	init_particle(domain, 5, n, T, &ion);

	auto J = make_form<Real, EDGE>(manifold);

	auto E = make_form<Real, EDGE>(manifold);

	auto B = make_form<Real, FACE>(manifold);

	E.clear();
	B.clear();

	Real t = 0;
	E.pull_back([=](typename manifold_type::coordinates_type const &x)
	{
		return nTuple<Real,3>(
				{	std::sin(x[1] *TWOPI),
					std::sin(x[2] *TWOPI),
					std::sin(x[0] *TWOPI)});
	});

	for (int i = 0; i < timestep; ++i)
	{

		J.clear();
		E += (curl(B) - J) * dt;
		B -= curl(E) * dt * 0.5;
		ion.next_timestep(dt, &J, E, B);
//		B -= curl(E) * dt * 0.5;

		LOGGER << save("/E", E, DataStream::SP_RECORD);
		LOGGER << save("/B", B, DataStream::SP_RECORD);
		LOGGER << save("/J", J, DataStream::SP_RECORD);
		t += dt;
	}

	save("/H", ion);

	GLOBAL_DATA_STREAM.close();
}

