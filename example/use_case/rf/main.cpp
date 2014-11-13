/**
 *  \file  RF.cpp
 *
 *  \date  2014-7-7
 *  \author salmon
 */

#include <iostream>

#include "../../src/utilities/log.h"
#include "../../src/io/data_stream.h"
#include "../../src/physics/constants.h"
#include "../../src/utilities/parse_command_line.h"
#include "../../src/parallel/message_comm.h"
#include "../../src/simpla_defs.h"

#include "../../src/mesh/geometry_cylindrical.h"
#include "../../src/mesh/mesh_rectangle.h"
#include "../../src/mesh/uniform_array.h"

#include "../../src/model/model.h"
#include "../../src/fetl/field.h"
#include "../../src/fetl/save_field.h"

#include "../../src/particle/particle_engine.h"
using namespace simpla;

static constexpr char key_words[] = "Cylindrical Geometry, Uniform Grid, single toridial model number, RF  ";

namespace simpla
{

/**
 * \ingroup ParticleEngine
 * \brief \f$\delta f\f$ engine
 */

typedef ParticleEngine<PolicyPICDeltaF> PICDeltaF;

template<>
struct ParticleEngine<PolicyPICDeltaF>
{
	typedef ParticleEngine<PolicyPICDeltaF> this_type;
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

	int J_at_the_center;

private:
	Real cmr_, q_kT_;
public:

	ParticleEngine() :
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

	~ParticleEngine()
	{
	}

	static std::string get_type_as_string()
	{
		return "DeltaF";
	}

	template<typename TJ, typename TE, typename TB>
	void next_timestep(Point_s * p, TJ* J, Real dt, TE const &fE, TB const & fB) const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

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

		J->scatter_cartesian(std::forward_as_tuple(p->x, p->v, p->f * charge * p->w));

	}

	static inline Point_s push_forward(coordinates_type const & x, Vec3 const &v, scalar_type f)
	{
		return std::move(Point_s( { x, v, f }));
	}

	static inline auto pull_back(Point_s const & p)
	DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))

}
;

} // namespace simpla

/**
 *   \example RF.cpp
 *    This is an example of interactive between rf wave and  plasma
 */

int main(int argc, char **argv)
{

	typedef FvMesh<CylindricalCoordinates<SurturedMesh>, true> mesh_type;

	typedef Model<mesh_type> model_type;

	typedef typename mesh_type::scalar_type scalar_type;

	static constexpr std::size_t   NDIMS = mesh_type::NDIMS;

	LOGGER.init(argc, argv);
	GLOBAL_COMM.init(argc,argv);
	GLOBAL_DATA_STREAM.init(argc,argv);

	model_type model;

	auto & mesh = model;

	nTuple<NDIMS, std::size_t  > dims = { 256, 256, 1 };

	Real dt = 1.0;

	std::size_t   num_of_step = 10;

	std::size_t   record_stride = 1;

	std::size_t   toridal_model_number = 1;

	bool just_a_test = false;

	std::string gfile = "";

	ParseCmdLine(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if( opt=="step")
		{
			num_of_step =ToValue<std::size_t  >(value);
		}
		else if(opt=="record")
		{
			record_stride =ToValue<std::size_t  >(value);
		}
		else if(opt=="i"||opt=="input")
		{
			gfile=(ToValue<std::string>(value));
		}
		else if(opt=="d"|| opt=="dims")
		{
			dims=ToValue<nTuple<3,std::size_t  >>(value);
		}
		else if(opt=="m"|| opt=="tordial_model_number")
		{
			toridal_model_number=ToValue<std::size_t  >(value);
		}
		else if( opt=="dt")
		{
			dt=ToValue<Real>(value);
		}
		else if(opt=="t")
		{
			just_a_test=true;
		}
		else if(opt=="V")
		{
			INFORM<<ShowShortVersion()<< std::endl
			<<"[Keywords: "<<key_words <<"]"<<std::endl;
			TheEnd(0);
		}

		else if(opt=="version")
		{
			INFORM<<ShowVersion()<< std::endl
			<<"[Keywords: "<<key_words <<"]"<<std::endl;
			TheEnd(0);
		}
		else if(opt=="help")
		{

			INFORM
			<< ShowCopyRight() << std::endl
			<<"[Keywords: "<<key_words <<"]"<<std::endl<<std::endl
			<<"Usage:"<<std::endl<<std::endl<<
			" -h        \t print this information\n"
			" -n<NUM>   \t number of steps\n"
			" -s<NUM>   \t recorder per <NUM> steps\n"
			" -o<STRING>\t output directory\n"
			" -i<STRING>\t configure file \n"
			" -c,--config <STRING>\t Lua script passed in as string \n"
			" -t        \t only read and parse input file, but do not process  \n"
			" -g,--generator   \t generator a demo input script file \n"
			" -v<NUM>   \t verbose  \n"
			" -V        \t print version  \n"
			" -q        \t quiet mode, standard out  \n"
			;
			TheEnd(0);
		}
		return CONTINUE;

	}

	);
//	if (!GLOBAL_DATA_STREAM.is_valid())
//	{
//		GLOBAL_DATA_STREAM.open_file("./");
//	}
//
	Particle<mesh_type, PICDeltaF> pp(model);

	INFORM << SIMPLA_LOGO << std::endl

	<< "Keyword: " << key_words << std::endl;

	LOGGER << "Pre-Process" << START;

//	if (gfile != "")
//	{
//		mesh.set_dimensions(dims);
//
//		mesh.set_dt(dt);
//
//		geqdsk.load(gfile);
//
//		geqdsk.SetUpModel(&model, toridal_model_number);
//
//		INFORM << geqdsk.save("/Input");
//	}
//	else
//	{
//		WARNING << ("No geqdsk-file is inputed!");
//
//		TheEnd(-1);
//	}

	INFORM << "Configuration: \n" << model;

	mesh_type::field<EDGE, scalar_type> E(model);
	mesh_type::field<FACE, scalar_type> B(model);
	mesh_type::field<VERTEX, nTuple<3, scalar_type>> u(model);
	mesh_type::field<VERTEX, scalar_type> Ti(model);
	mesh_type::field<VERTEX, scalar_type> Te(model);
	mesh_type::field<VERTEX, scalar_type> n(model);
	mesh_type::field<EDGE, scalar_type> J(model);
	mesh_type::field<EDGE, scalar_type> p(model);

//	auto limiter_face = model.SelectInterface(FACE, model_type::VACUUM, model_type::NONE);
//	auto limiter_edge = model.SelectInterface(EDGE, model_type::VACUUM, model_type::NONE);
//
//	geqdsk.GetProfile("ne", &n);
//	geqdsk.GetProfile("pres", &p);
//	geqdsk.GetProfile("Ti", &Ti);
//	geqdsk.GetProfile("Ti", &Te);
//	geqdsk.GetProfile("B", &B);

	INFORM << SINGLELINE;

	LOGGER << "Process " << START;

	TheStart();

	if (just_a_test)
	{
		LOGGER << "Just test configure files";
	}
	else
	{
		GLOBAL_DATA_STREAM.cd("/save");
		GLOBAL_DATA_STREAM.property("compact storage",true);

		//   save initial value
		LOGGER << SAVE(B);
		LOGGER << SAVE(n);
		LOGGER << SAVE(p);
		LOGGER << SAVE(u);
		LOGGER << SAVE(Ti);
		LOGGER << SAVE(J);

		for (int i = 0; i < num_of_step; ++i)
		{

			mesh.next_timestep();

			DEFINE_PHYSICAL_CONST

			INFORM << "[" <<mesh. get_clock() << "]"

			<< "Simulation Time = " << (mesh.get_time() / CONSTANTS["s"]) << "[s]";

			// boundary constraints
//
//			{
//				for(auto s:limiter_face)
//				{
//					get_value(B,s)=0.0;
//				}
//
//				for(auto s:limiter_edge)
//				{
//					get_value(J,s)=0.0;
//				}
//			}

			if (i % record_stride == 0)
			{
				LOGGER << SAVE(B);
				LOGGER << SAVE(n);
				LOGGER << SAVE(p);
				LOGGER << SAVE(u);
				LOGGER << SAVE(Ti);
				LOGGER << SAVE(J);

			}
		}

	}
	LOGGER << "Process" << DONE;

	LOGGER << "Post-Process" << START;

	INFORM << "OutPut Path:" << GLOBAL_DATA_STREAM.pwd();

	LOGGER << "Post-Process" << DONE;

	GLOBAL_DATA_STREAM.close();
	GLOBAL_COMM.close();
	TheEnd();

}
