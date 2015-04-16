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
#include "../utilities/config_parser.h"
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

	ConfigParser options;
	options.init(argc, argv);

	bool no_logo = false;
	bool show_help = false;

	if (options["V"] || options["version"])
	{
		MESSAGE << "SIMPla " << ShowVersion();
		TheEnd(0);
		return TERMINATE;
	}
	else if (options["h"] || options["help"])
	{
		show_help = true;
	}

	SpAppList & applist = SingletonHolder<SpAppList>::instance();

	if (GLOBAL_COMM.process_num() == 0)

	MESSAGE << ShowCopyRight() << endl;

	if (options["SHOW_HELP"])
	{
		MESSAGE << " Usage: " << argv[0]
				<< " --case <id of use case>  <options> ..." << endl << endl;

		MESSAGE << " Use case list:" << std::endl;

		for (auto const & item : applist)
		{
			MESSAGE << "\t" << item.first << "\t:" << item.second->description()
					<< std::endl;
		}

		MESSAGE << " Options:" << endl;

		SHOW_OPTIONS("-h", "Print help information");
		SHOW_OPTIONS("-v,--version", "Print version");
		SHOW_OPTIONS("-g,--generator", "Generates  demo configure file");
	}

	auto item = applist.begin();

	if (options["case"])
	{
		item = applist.find(options["case"].template as<std::string>());
	}

	if (item != applist.end())
	{
		GLOBAL_COMM.barrier();

		MESSAGE
		<<std::endl
		<<"===================================================="<<std::endl
		<<"   Use Case ["<<item->first <<"]:  "<<std::endl
		<<"\t"<<item->second->description()<<std::endl
		<<"----------------------------------------------------"<<std::endl
		;

		item->second->body(options);

		GLOBAL_COMM.barrier();
	}

	MESSAGE << "===================================================="
			<< std::endl

			<< "\t >>> Done <<< " << std::endl;
	close_io();
	close_parallel();
	close_logger();

	return 0;

}
