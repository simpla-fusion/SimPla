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

	std::string help_message;

	help_message += init_logger(argc, argv);
	help_message += init_parallel(argc, argv);
	help_message += init_io(argc, argv);

	ConfigParser options;
	SpAppList & applist = SingletonHolder<SpAppList>::instance();

	help_message += options.init(argc, argv);

	if (GLOBAL_COMM.process_num() == 0)
	{	MESSAGE << ShowCopyRight() << endl;}

	if (options["V"] || options["version"])
	{
		MESSAGE << "SIMPla " << ShowVersion();
		TheEnd(0);
		return TERMINATE;
	}
	else if (options["h"] || options["help"])
	{

		MESSAGE << " Usage: " << argv[0]
				<< " --case <id of use case>  <options> ..." << endl << endl;

		MESSAGE << " Use case list:" << std::endl;

		for (auto const & item : applist)
		{
			MESSAGE << "\t" << item.first << "\t:" << item.second->description()
					<< std::endl;
		}

		MESSAGE << " Options:" << endl

		<< "\t -h \t, Print help information \n"
				"\t -v,--version \t, Print version \n"
				"\t -g,--generator\t, Generates  demo configure file \n"
		<< help_message;
		TheEnd(0);
		return TERMINATE;

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
