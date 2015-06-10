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

	help_message += logger::init_logger(argc, argv);
	help_message += init_parallel(argc, argv);
	help_message += init_io(argc, argv);

	ConfigParser options;
	SpAppList & applist = SingletonHolder<SpAppList>::instance();

	help_message += options.init(argc, argv);

	if (GLOBAL_COMM.process_num() == 0)
	{	MESSAGE << ShowCopyRight() << std::endl;}

	if (options["V"] || options["version"])
	{
		MESSAGE<< "SIMPla " << ShowVersion();
		TheEnd(0);
		return TERMINATE;
	}
	else if (options["h"] || options["help"])
	{

		MESSAGE

		<< " Usage: " << argv[0] << "   <options> ..." << std::endl << std::endl;

		MESSAGE

		<< " Options:" <<std:: endl

		<<

		"\t -h,\t--help            \t, Print a usage message and exit.\n"
		"\t -v,\t--version         \t, Print version information exit. \n"
//						"\t -g,\t--generator       \t, Generates a demo configure file \n"
		"\n"
		"\t--case <CASE ID>         \t, Select a case <CASE ID> to execute \n "
		"\t--case_help <CASE ID>    \t, Print a usag message of case <CASE ID> \n "
		<< help_message;

		MESSAGE

		<< " Use case list:" << std::endl

		<< "        CASE ID     | Description " << std::endl
		<< "--------------------|-------------------------------------"
		<<std:: endl;

		for (auto const & item : applist)
		{

			MESSAGE << std::setw(19) << std::right << item.first << " |"
			<< item.second->description() <<std:: endl;
		}

		TheEnd(0);

	}
	{
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

		MESSAGE<< "===================================================="
		<< std::endl

		<< "\t >>> Done <<< " << std::endl;
	}
	close_io();
	close_parallel();
	logger::close_logger();

	return 0;

}
