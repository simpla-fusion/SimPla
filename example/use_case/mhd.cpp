/**
 *  \file  mhd.cpp
 *
 *  \date  2014-7-7
 *  \author salmon
 */

#include <iostream>

#include "../src/utilities/log.h"
#include "../src/io/data_stream.h"
#include "../src/physics/constants.h"
#include "../src/utilities/parse_command_line.h"
#include "../src/parallel/message_comm.h"
#include "../src/simpla_defs.h"

#include "../src/mesh/geometry_cylindrical.h"
#include "../src/mesh/mesh_rectangle.h"
#include "../src/mesh/uniform_array.h"

#include "../src/model/model.h"
#include "../src/model/geqdsk.h"

#include "../src/fetl/fetl.h"
#include "../src/fetl/field_ops.h"
#include "../src/fetl/save_field.h"

using namespace simpla;

static constexpr char key_words[] = "Cylindrical Geometry, Uniform Grid, single toridal model number  ";

/**
 *   \example  mhd/mhd.cpp
 *    This is an example of ideal MHD instability, ELM
 */

int main(int argc, char **argv)
{

	typedef Mesh<CylindricalGeometry<UniformArray>, true> mesh_type;

	typedef Model<mesh_type> model_type;

	typedef typename mesh_type::scalar_type scalar_type;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	LOG_STREAM.init(argc,argv);
	GLOBAL_COMM.init(argc,argv);
	GLOBAL_DATA_STREAM.init(argc,argv);

	GEqdsk geqdsk;

	model_type model;

	nTuple<NDIMS, size_t> dims = { 256, 256, 1 };

	Real dt = 1.0;

	size_t num_of_step = 10;

	size_t record_stride = 1;

	unsigned int toridal_model_number = 1;

	bool just_a_test = false;

	std::string gfile = "";

	ParseCmdLine(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if( opt=="step")
		{
			num_of_step =ToValue<size_t>(value);
		}
		else if(opt=="record")
		{
			record_stride =ToValue<size_t>(value);
		}
		else if(opt=="i"||opt=="input")
		{
			gfile=(ToValue<std::string>(value));
		}
		else if(opt=="d"|| opt=="dims")
		{
			dims=ToValue<nTuple<3,size_t>>(value);
		}
		else if(opt=="m"|| opt=="tordial_model_number")
		{
			toridal_model_number=ToValue<unsigned int>(value);
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
	if (!GLOBAL_DATA_STREAM.is_ready())
	{
		GLOBAL_DATA_STREAM.open_file("./");
	}

	INFORM << SIMPLA_LOGO << std::endl

	<< "Keyword: " << key_words << std::endl;

	LOGGER << "Pre-Process" << START;

	if (gfile != "")
	{
		model.set_dimensions(dims);

		model.set_dt(dt);

		geqdsk.load(gfile);

		typename mesh_type::coordinates_type src_min;
		typename mesh_type::coordinates_type src_max;

		std::tie(src_min, src_max) = geqdsk.get_extents();

		typename mesh_type::coordinates_type min;
		typename mesh_type::coordinates_type max;

		std::tie(min, max) = model.get_extents();

		min[(mesh_type::ZAxis + 2) % 3] = src_min[GEqdsk::RAxis];
		max[(mesh_type::ZAxis + 2) % 3] = src_max[GEqdsk::RAxis];

		min[mesh_type::ZAxis] = src_min[GEqdsk::ZAxis];
		max[mesh_type::ZAxis] = src_max[GEqdsk::ZAxis];

		model.set_extents(min, max);

		geqdsk.SetUpMaterial(&model, toridal_model_number);

		INFORM << geqdsk.save("/Input");
	}
	else
	{
		WARNING << ("No geqdsk-file is inputed!");

		TheEnd(-1);
	}

	model.Update();

	INFORM << "Configuration: \n" << model;

	auto E = model.template make_field<EDGE, scalar_type>();
	E.clear();

	auto B = model.template make_field<EDGE, scalar_type>();
	B.clear();

	auto dE = model.template make_field<EDGE, scalar_type>();
	dE.clear();

	auto dB = model.template make_field<EDGE, scalar_type>();
	dB.clear();

	auto J0 = model.template make_field<EDGE, scalar_type>();
	J0.clear();

	auto Jext = model.template make_field<EDGE, scalar_type>();
	Jext.clear();

	auto u = model.template make_field<VERTEX, nTuple<3, scalar_type>>();
	u.clear();

	auto T = model.template make_field<VERTEX, scalar_type>();
	T.clear();

	auto n = model.template make_field<VERTEX, scalar_type>();
	n.clear();

	auto J = model.template make_field<EDGE, scalar_type>();
	J.clear();

	auto p = model.template make_field<VERTEX, scalar_type>();
	p.clear();

	auto limiter_face = model.SelectInterface(FACE, model_type::VACUUM, model_type::NONE);

	auto limiter_edge = model.SelectInterface(EDGE, model_type::VACUUM, model_type::NONE);

	geqdsk.GetProfile("ne", &n);
	geqdsk.GetProfile("pres", &p);
	geqdsk.GetProfile("Ti", &T);
	geqdsk.GetProfile("B", &B);

	INFORM << SINGLELINE;

	LOGGER << "Process " << START;

	TheStart();

	if (just_a_test)
	{
		LOGGER << "Just test configure files";
	}
	else
	{
		GLOBAL_DATA_STREAM.open_group("/save");
		GLOBAL_DATA_STREAM.EnableCompactStorable();

		//   save initial value
		LOGGER << SAVE(B);
		LOGGER << SAVE(n);
		LOGGER << SAVE(p);
		LOGGER << SAVE(u);
		LOGGER << SAVE(T);
		LOGGER << SAVE(J);

		DEFINE_PHYSICAL_CONST

		for (int i = 0; i < num_of_step; ++i)
		{

			LOG_CMD(Jext = J0);

			LOG_CMD(B += dB * 0.5);	//  B(t=0 -> 1/2)

			dE.clear();
			LOG_CMD(dE += Curl(B)/(mu0 * epsilon0) *dt);

			LOG_CMD(dE -= Jext * (dt / epsilon0));

			dE = Jext * (dt / epsilon0);
			//   particle 1/2 -> 1  . To n[1/2], J[1/2]

			LOG_CMD(E += dE);// E(t=0 -> 1)

			dB.clear();
			LOG_CMD( dB -= Curl(E)*dt);

			LOG_CMD(B += dB * 0.5);//	B(t=1/2 -> 1)

			model.next_timestep();

			DEFINE_PHYSICAL_CONST

			INFORM << "[" <<model. get_clock() << "]"

			<< "Simulation Time = " << (model.get_time() / CONSTANTS["s"]) << "[s]";

			// todo Next time step

			// boundary constraints

			{
				for(auto s:limiter_face)
				{
					get_value(B,s)=0.0;
				}

				for(auto s:limiter_edge)
				{
					get_value(J,s)=0.0;
				}
			}

			if (i % record_stride == 0)
			{
				LOGGER << SAVE(B);
				LOGGER << SAVE(n);
				LOGGER << SAVE(p);
				LOGGER << SAVE(u);
				LOGGER << SAVE(T);
				LOGGER << SAVE(J);

			}
		}

		GLOBAL_DATA_STREAM.DisableCompactStorable();
	}
	LOGGER << "Process" << DONE;

	LOGGER << "Post-Process" << START;

	INFORM << "OutPut Path:" << GLOBAL_DATA_STREAM.GetCurrentPath();

	LOGGER << "Post-Process" << DONE;

	GLOBAL_DATA_STREAM.Close();
	GLOBAL_COMM.Close();
	TheEnd();

}
