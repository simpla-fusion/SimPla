/*
 * ggauge.cpp
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#include "include/simpla_defs.h"
#include "utilities/properties.h"
#include "engine/context.h"
#include "engine/detail/context_impl.h"

#include "fetl/grid/uniform_rect.h"

using namespace simpla;

int main(int argc, char **argv)
{

	Context<UniformRectGrid> ctx;

	size_t max_step = 1000;

	size_t record_stride = 1;

	std::string input = "simpla.xml";
	std::string output = "untitled";
	std::string log_file = "simpla.log";

	for (int i = 1; i < argc; ++i)
	{
		switch (argv[i][1])
		{
		case 'n':
			max_step = atoi(argv[i] + 2);
			break;
		case 's':
			record_stride = atoi(argv[i] + 2);
			break;
		case 'o':
			output = argv[i] + 2;
			break;
		case 'i':
			input = argv[i] + 2;
			break;
		case 'l':
			log_file =  std::string(argv[i] + 2);
			break;
		case 'v':
			Log::Verbose(atof(argv[i] + 2));
			break;
		}

	}

	Log::OpenFile(output + "/" +log_file);

	ptree pt;

	read_file(input, pt);

	ctx.output_path = output;


	ctx.Parse(pt.get_child("Context"));


	INFORM

	<< std::endl

	<< DOUBLELINE << std::endl

	<< SIMPLA_LOGO << std::endl

	<< DOUBLELINE << std::endl

	<< std::setw(20) << "Teimstamp : " << Log::Teimstamp() << std::endl

	<< std::setw(20) << "Num. of procs. : " << omp_get_num_procs() << std::endl

	<< std::setw(20) << "Num. of threads : " << omp_get_max_threads()

	<< std::endl

	<< std::setw(20) << "Configure File : " << input << std::endl

	<< std::setw(20) << "Output Path : " << ctx.output_path << std::endl

	<< std::setw(20) << "Log File : " << log_file << std::endl

	<< std::setw(20) << "Number of steps : " << max_step << std::endl

	<< std::setw(20) << "Record/steps : " << record_stride << std::endl

	<< SINGLELINE << std::endl;

	INFORM

	<< ctx.Summary() << std::endl

	<< SINGLELINE << std::endl

	<< std::endl;

	INFORM << "====== Start PreProcess! =======" << std::endl;

	ctx.PreProcess();

	INFORM << "====== Start Process! =======" << std::endl;

	ctx.Process();

	INFORM << "====== Done! =======" << std::endl;
}

