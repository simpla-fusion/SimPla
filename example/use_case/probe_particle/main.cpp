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

	template<typename TE, typename TB>
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

//	Particle<manifold_type, PICDemo, PolicyProbeParticle>

	ProbeParticle<manifold_type, PICDemo> ion(manifold, mass, charge, Te);

	INFORM << "=========================" << std::endl;
	INFORM << "dt =  \t" << dt << std::endl;
	INFORM << "time step =  \t" << timestep << std::endl;
	INFORM << "ion =  \t {" << ion << "}" << std::endl;
	INFORM << "=========================" << std::endl;

	PICDemo::Point_s p =
	{ 0, 0, 0, 1, 0, 0, 1, 0 };

	ion.push_back(p);

	auto n = [](typename manifold_type::coordinates_type const & x )
	{	return 2.0;};

	auto T = [](typename manifold_type::coordinates_type const & x )
	{	return 1.0;};

	init_particle(make_domain<VERTEX>(manifold), 5, n, T, &ion);

	auto B = [](nTuple<Real,3> const & )
	{
		return nTuple<Real,3>(
				{	0,0,2});
	};
	auto E = [](nTuple<Real,3> const & )
	{
		return nTuple<Real,3>(
				{	0,0,2});
	};

	ion.next_timestep(timestep, dt, E, B);

	//	for (int i = 0; i < timestep; ++i)
//	{
//		PICDemo::next_timestep(&p, dt, E, B);
////		INFORM << save("/H", ion, DataStream::SP_CACHE | DataStream::SP_RECORD);
//	}

	GLOBAL_DATA_STREAM.close();
}

