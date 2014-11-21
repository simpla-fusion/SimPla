/*
 * use_case_main.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include <iostream>
#include <string>

#include "../io/io.h"
#include "../parallel/parallel.h"
#include "../simpla_defs.h"
#include "../utilities/log.h"
#include "../utilities/parse_command_line.h"
#include "use_case.h"
#include "logo.h"

int main(int argc, char **argv)
{
	using namespace simpla;

	bool no_logo = false;
	bool show_help = false;

	parse_cmd_line(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if(opt=="V" || opt=="version")
		{
			STDOUT<<"simpla "<< ShowVersion();
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

	STDOUT << ShowCopyRight() << std::endl;

	if (show_help)
	{
		STDOUT << " Usage: " << argv[0] << "  <options> ..." << std::endl
				<< std::endl;

		STDOUT << " Options:" << std::endl;

		SHOW_OPTIONS("-h", "print this information");
		SHOW_OPTIONS("-v,--version", "print version");
	}
	init_logger(argc, argv);
	init_parallel(argc, argv);
	init_io(argc, argv);

	RunAllUseCase(argc, argv);

	close_io();
	close_parallel();
	close_logger();

	return 0;

}

