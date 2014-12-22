/*
 * use_case_main.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include <iostream>
#include <string>

#include "../io/io.h"
#include "../simpla_defs.h"
#include "../parallel/parallel.h"
#include "../utilities/log.h"
#include "../utilities/parse_command_line.h"
#include "use_case.h"
#include "logo.h"

int main(int argc, char **argv)
{
	using namespace simpla;


	init_logger(argc, argv);
	init_parallel(argc, argv);
	init_io(argc, argv);


	bool no_logo = false;
	bool show_help = false;

	parse_cmd_line(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if(opt=="V" || opt=="version")
		{
			MESSAGE<<"simpla "<< ShowVersion();
			TheEnd(0);
			return TERMINATE;
		}
		else if(opt=="h"||opt=="help")
		{
			show_help=true;
		}
		return CONTINUE;
	}

	);
	if (GLOBAL_COMM.get_rank() == 0)
		MESSAGE << ShowCopyRight() << std::endl;

	if (show_help)
	{
		MESSAGE << " Usage: " << argv[0] << "  <options> ..." << std::endl
				<< std::endl;

		MESSAGE << " Options:" << std::endl;

		SHOW_OPTIONS("-h", "Print help information");
		SHOW_OPTIONS("-v,--version", "Print version");
		SHOW_OPTIONS("-g,--generator", "Generates  demo configure file");
	}

	RunAllUseCase(argc, argv);

	close_io();
	close_parallel();
	close_logger();
	return 0;

}

