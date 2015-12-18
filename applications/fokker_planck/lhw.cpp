/**
 * \file lhw.cpp
 *
 * @date    2014-7-30  AM7:20:34
 * @author salmon
 */

#include <random>
#include <string>
#include <functional>
#include <omp.h>

#include "../../src/io/DataStream.h"
#include "../../src/simpla_defs.h"
#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "../../src/utilities/parse_command_line.h"
#include "../../src/utilities/utilities.h"
#include "../../src/gtl/ntuple.h"
#include "../../src/utilities/ntuple_noet.h"
#include "../../src/physics/PhysicalConstants.h"
#include "../../src/io/hdf5_datatype.h"

#include "../../src/numeric/multi_normal_distribution.h"
#include "../../src/numeric/rectangle_distribution.h"

static const char descritpion[] = "Example for fokker_planck";
static const char sub_version[] = __FILE__ "(version: 0.0.1)";
static constexpr std::size_t   NDIMS = 3;

using namespace simpla;

struct PICDeltaF
{

public:
	typedef nTuple<3, Real> coordinate_tuple;
	typedef std::complex<Real> scalar_type;

	typedef PICDeltaF this_type;

	struct Point_s
	{
		coordinate_tuple x;
		Vec3 v;
		Real f;
		scalar_type w;

	};

	Real m;
	Real q;
	Real T;

private:
	Real cmr_, q_kT_;
public:
	PICDeltaF()
			: m(1.0), q(1.0), T(1.0)
	{

		load();
	}

	void load()
	{
		DEFINE_PHYSICAL_CONST

		q_kT_ = q / (boltzmann_constant * T);
		cmr_ = q / m;
		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "    H5T_ARRAY { [2] H5T_NATIVE_DOUBLE}     \"w\" : " << (offsetof(Point_s, w)) << ";"

			<< "}";

			GLOBAL_HDF5_DATA_TYPE_FACTORY. Register< Point_s > (os.str());
		}

	}
	~PICDeltaF()
	{
	}

	template<typename TE,typename TB>
	inline void next_timestep (Point_s * p, Real dt, TE const &fE0, TB const & fB0 )const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB0( p->x);
		auto E = fE0( p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_;
		auto a = (-Dot(E, p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->v += v_;
		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

	}

};

int main(int argc, char **argv)
{
	LOGGER.init(argc, argv);
	GLOBAL_DATA_STREAM.init(argc, argv);

	typedef typename PICDeltaF::coordinate_tuple coordinate_tuple;

	PICDeltaF engine;

	std::vector<PICDeltaF::Point_s> particles(10);

	Real dt = 1.0;

	Real k1 = 0.0;
	Real k2 = 0.0;

	LuaObject dict;

	std::string context_type = "";

	std::size_t   num_of_step = 10;

	std::size_t   record_stride = 1;

	bool just_a_test = false;

	parse_cmd_line(argc, argv, [&](std::string const & opt,std::string const & value)->int
	{
		if(opt=="n"||opt=="num_of_step")
		{
			num_of_step =string_to_value<std::size_t  >(value);
		}
		else if(opt=="s"||opt=="record_stride")
		{
			record_stride =string_to_value<std::size_t  >(value);
		}
		else if(opt=="i"||opt=="input")
		{
			dict.parse_file(value);
		}
		else if(opt=="c"|| opt=="command")
		{
			dict.parse_string(value);
		}
		else if(opt=="g"|| opt=="generator")
		{
			INFORM
			<< ShowCopyRight() << std::endl
			<< "Too lazy to implemented it\n"<< std::endl;
			TheEnd(1);
		}
		else if(opt=="t")
		{
			just_a_test=true;
		}
		else if(opt=="V")
		{
			INFORM<<ShowShortVersion()<< sub_version<< std::endl;
			TheEnd(0);
		}
		else if(opt=="version")
		{
			INFORM<<ShowVersion()<< sub_version<<std::endl;
			TheEnd(0);
		}
		else if(opt=="help")
		{
			INFORM
			<< ShowCopyRight() << std::endl

			<< descritpion;

			TheEnd(0);
		}

		return CONTINUE;

	}

	);

	INFORM << SIMPLA_LOGO

	<< " C++:" << __cplusplus << std::endl

	<< "App:" << sub_version << std::endl;

	LOGGER << "Parse Command Line." << DONE;

	INFORM << SINGLELINE;

	LOGGER << "Pre-Process" << START;

	// Main Loop ============================================

	LOGGER << "Process " << START;

	TheStart();

	auto B = [](coordinate_tuple const &)
	{
		return Vec3(
				{	0,0,1.0});
	};

	auto E = [k1,k2](coordinate_tuple const & x)
	{
		return Vec3(
				{	0,std::sin(k1*x[1]),std::sin(k2*x[2])});
	};

	if (just_a_test)
	{
		LOGGER << "Just test configure files";
	}
	else
	{
		GLOBAL_DATA_STREAM.set_property< int>("Cache Depth", 20);

		GLOBAL_DATA_STREAM.set_property("Force Record Storage",true);
		GLOBAL_DATA_STREAM.set_property("Force Write Cache",true);

		DEFINE_PHYSICAL_CONST;

#pragma omp parallel
		{
			std::mt19937 rnd_gen(NDIMS * 2);

			std::size_t   num_of_particles=particles.size();

			int num_threads=omp_get_num_threads();
			int thread_num=omp_get_thread_num();

			std::size_t   ib= num_of_particles*thread_num/num_threads;

			std::size_t   ie= (num_of_particles+1)*thread_num/num_threads;

			rnd_gen.discard(ib*NDIMS*2);

			nTuple<3, Real> x, v;

			Real inv_sample_density = 1.0;

			rectangle_distribution<NDIMS> x_dist;

			multi_normal_distribution<NDIMS> v_dist(boltzmann_constant * engine.T / engine.m );

			for (int i = ib; i < ie; ++i)
			{
				x_dist(rnd_gen, &(particles[i].x[0]));

				v_dist(rnd_gen, &(particles[i].v[0]));

				particles[i].f=1.0;
				particles[i].w=0.0;
			}

		}
		for (int step = 0; step < num_of_step; ++step)
		{
			LOGGER << "STEP: " << step;

			int ie=particles.size();
#pragma omp parallel for
			for(int i=0;i<ie;++i)
			{
				engine.next_timestep(&particles[i],dt,E,B);
			}

			if (step % record_stride == 0)
			{
				LOGGER<<SAVE(particles);
			}
		}
		GLOBAL_DATA_STREAM.command("Flush");
		GLOBAL_DATA_STREAM.set_property("Force Write Cache",false);
		GLOBAL_DATA_STREAM.set_property("Force Record Storage",false);
	}
	LOGGER << "Process" << DONE;

	INFORM << SINGLELINE;

	LOGGER << "Post-Process" << START;

	INFORM << "OutPut Path:" << GLOBAL_DATA_STREAM.pwd();

	LOGGER << "Post-Process" << DONE;

	INFORM << SINGLELINE;
	GLOBAL_DATA_STREAM.close();

	TheEnd();
}

