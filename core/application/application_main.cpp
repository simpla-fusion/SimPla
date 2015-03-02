/*
 * application_main.cpp
 *
 *  Created on: 2015年1月7日
 *      Author: salmon
 */

#include <iostream>
#include <string>

#include "../io/io.h"
#include "../parallel/parallel.h"
#include "../sp_config.h"
#include "../utilities/log.h"
#include "../utilities/parse_command_line.h"
#include "application.h"
#include "logo.h"

/**
 *  @ingroup application
 *
 *  main entry of user case.
 */
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
			MESSAGE<<"SIMPla "<< ShowVersion();
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
	if (GLOBAL_COMM.process_num() == 0)

	MESSAGE << ShowCopyRight() << endl;

	if (show_help)
	{
		MESSAGE << " Usage: " << argv[0] << "  <options> ..." << endl << endl;

		MESSAGE << " Options:" << endl;

		SHOW_OPTIONS("-h", "Print help information");
		SHOW_OPTIONS("-v,--version", "Print version");
		SHOW_OPTIONS("-g,--generator", "Generates  demo configure file");
	}

	MESSAGE << "--------- START --------- " << endl;

	run_all_apps(argc, argv);

	MESSAGE << "--------- DONE --------- " << endl;

	close_io();
	close_parallel();
	close_logger();

	return 0;

}
