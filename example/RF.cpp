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
#include "../../src/model/geqdsk.h"

#include "../../src/fetl/fetl.h"
#include "../../src/fetl/save_field.h"

using namespace simpla;

static constexpr char key_words[] = "Cylindrical Geometry, Uniform Grid, single toridal model number, RF  ";

/**
 *   \example RF.cpp
 *    This is an example of interactive between rf wave and  plasma
 */

int main(int argc, char **argv)
{

	typedef Mesh<CylindricalGeometry<UniformArray>, true> cylindrical_mesh;

	typedef Model<cylindrical_mesh> model_type;

	typedef typename cylindrical_mesh::scalar_type scalar_type;

	static constexpr unsigned int NDIMS = cylindrical_mesh::NDIMS;

	LOG_STREAM.Init(argc,argv);
	GLOBAL_COMM.Init(argc,argv);
	GLOBAL_DATA_STREAM.Init(argc,argv);

	GEqdsk geqdsk;

	model_type model;

	auto & mesh = model.mesh;

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
		GLOBAL_DATA_STREAM.OpenFile("./");
	}

	INFORM << SIMPLA_LOGO << std::endl

	<< "Keyword: " << key_words << std::endl;

	LOGGER << "Pre-Process" << START;

	if (gfile != "")
	{
		mesh.set_dimensions(dims);

		mesh.set_dt(dt);

		geqdsk.load(gfile);

		geqdsk.SetUpModel(&model, toridal_model_number);

		INFORM << geqdsk.save("/Input");
	}
	else
	{
		WARNING << ("No geqdsk-file is inputed!");

		TheEnd(-1);
	}

	INFORM << "Configuration: \n" << model;

	auto E = mesh.template make_field<EDGE, scalar_type>();
	E.clear();

	auto B = mesh.template make_field<FACE, scalar_type>();
	B.clear();

	auto u = mesh.template make_field<VERTEX, nTuple<3, scalar_type>>();
	u.clear();

	auto Ti = mesh.template make_field<VERTEX, scalar_type>();
	Ti.clear();

	auto Te = mesh.template make_field<VERTEX, scalar_type>();
	Te.clear();

	auto n = mesh.template make_field<VERTEX, scalar_type>();
	n.clear();

	auto J = mesh.template make_field<EDGE, scalar_type>();
	J.clear();

	auto p = mesh.template make_field<VERTEX, scalar_type>();
	p.clear();

	auto limiter_face = model.SelectInterface(FACE, model_type::VACUUM, model_type::NONE);

	auto limiter_edge = model.SelectInterface(EDGE, model_type::VACUUM, model_type::NONE);

	geqdsk.GetProfile("ne", &n);
	geqdsk.GetProfile("pres", &p);
	geqdsk.GetProfile("Ti", &Ti);
	geqdsk.GetProfile("Ti", &Te);
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
		GLOBAL_DATA_STREAM.OpenGroup("/save");
		GLOBAL_DATA_STREAM.EnableCompactStorable();

		//   save initial value
		LOGGER << SAVE(B);
		LOGGER << SAVE(n);
		LOGGER << SAVE(p);
		LOGGER << SAVE(u);
		LOGGER << SAVE(T);
		LOGGER << SAVE(J);

		for (int i = 0; i < num_of_step; ++i)
		{

			mesh.next_timestep();

			DEFINE_PHYSICAL_CONST

			INFORM << "[" <<mesh. get_clock() << "]"

			<< "Simulation Time = " << (mesh.get_time() / CONSTANTS["s"]) << "[s]";

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
