/**
 *  \file  mhd.cpp
 *
 *  \date  2014-7-7
 *  \author salmon
 */
/**
 *   \example  mhd/mhd.cpp
 *    This is an example of ideal MHD instability, ELM
 */

#include <iostream>

#include "../../src/mesh/geometry_cylindrical.h"
#include "../../src/mesh/mesh_rectangle.h"
#include "../../src/mesh/uniform_array.h"
#include "../../src/fetl/fetl.h"
#include "../../src/fetl/save_field.h"
#include "../../src/model/model.h"
#include "../../src/model/geqdsk.h"
#include "../../src/physics/constants.h"
#include "../../src/utilities/parse_command_line.h"

#include "../../src/utilities/log.h"
#include "../../src/io/data_stream.h"
#include "../../src/parallel/message_comm.h"

#include "../../src/simpla_defs.h"

using namespace simpla;

static constexpr char key_words[] = "Cylindrical Geometry, Uniform Grid, single toridal model number  ";

int main(int argc, char **argv)
{

	LOG_STREAM.Init(argc,argv);
	GLOBAL_COMM.Init(argc,argv);
	GLOBAL_DATA_STREAM.Init(argc,argv);

	typedef Mesh<CylindricalGeometry<UniformArray>, true> cylindrical_mesh;

	static constexpr unsigned int NDIMS = cylindrical_mesh::NDIMS;

	GEqdsk geqdsk;

	Model<cylindrical_mesh> model;

	cylindrical_mesh & mesh=model.mesh;

	nTuple<NDIMS, size_t> dims =
	{	256, 256, 1};

	size_t num_of_step = 10;

	size_t record_stride = 1;

	unsigned int toridal_model_number = 1;

	bool just_a_test = false;

	std::string gfile="";

	ParseCmdLine(argc, argv,

			[&](std::string const & opt,std::string const & value)->int
			{
				if(opt=="n"||opt=="num_of_step")
				{
					num_of_step =ToValue<size_t>(value);
				}
				else if(opt=="s"||opt=="record_stride")
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
	if (!GLOBAL_DATA_STREAM.IsOpened())
	{
		GLOBAL_DATA_STREAM.OpenFile("./");
	}

	INFORM << SIMPLA_LOGO<<std::endl

	<< "Keyword: "<<key_words << std::endl;

	INFORM << SINGLELINE;

	LOGGER << "Pre-Process" << START;

	if(gfile!="")
	{
		mesh.set_dimensions(dims);

		geqdsk.Read(gfile);

		geqdsk.SetUpModel(&model);

		INFORM << geqdsk.Save("/Input");
	}
	else
	{
		ERROR("No geqdsk-file is inputed!");
	}

	INFORM <<"Configuration: \n" << model;

	INFORM << SINGLELINE;

	LOGGER << "Process " << START;

	TheStart();

	if (just_a_test)
	{
		LOGGER << "Just test configure files";
	}
	else
	{

		GLOBAL_DATA_STREAM.EnableCompactStorable();

		// todo save initial value
		//	LOGGER << SAVE();

		for (int i = 0; i < num_of_step; ++i)
		{
			LOGGER << "STEP: " << i;

			// todo Next time step

			mesh.NextTimeStep();

			if (i % record_stride == 0)
			{
				// todo dump data
			}
		}

		GLOBAL_DATA_STREAM.DisableCompactStorable();
	}
	LOGGER << "Process" << DONE;

	INFORM << SINGLELINE;

	LOGGER << "Post-Process" << START;

	INFORM << "OutPut Path:" << GLOBAL_DATA_STREAM.GetCurrentPath();

	LOGGER << "Post-Process" << DONE;

	INFORM << SINGLELINE;
	GLOBAL_DATA_STREAM.Close();
	GLOBAL_COMM.Close();
	TheEnd();

}
