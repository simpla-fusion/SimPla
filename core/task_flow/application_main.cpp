/**
 * @file application_main.cpp
 *
 *  Created on: 2015-1-7
 *      Author: salmon
 */

#include <iostream>
#include <string>

#include "application.h"
#include "../io/io.h"
#include "../parallel/Parallel.h"
#include "../sp_config.h"
#include "../gtl/utilities/log.h"
#include "../gtl/utilities/config_parser.h"
#include "../gtl/utilities/logo.h"

/**
 *  @ingroup task_flow
 *
 *  main entry of user case.
 */
int main(int argc, char **argv)
{
    using namespace simpla;


    logger::init(argc, argv);

    parallel::init(argc, argv);

    io::init(argc, argv);


    ConfigParser options;

    SpAppList &applist = SingletonHolder<SpAppList>::instance();

    options.init(argc, argv);

    if (GLOBAL_COMM.process_num() == 0) { MESSAGE << ShowCopyRight() << std::endl; }

    if (options["V"] || options["version"])
    {
        MESSAGE << "SIMPla " << ShowVersion();
        TheEnd(0);
        return TERMINATE;
    }
    else if (options["h"] || options["help"])
    {

        MESSAGE

        << " Usage: " << argv[0] << "   <options> ..." << std::endl << std::endl;

        MESSAGE

        << " Options:" << std::endl

        <<

        "\t -h,\t--help            \t, Print a usage message and exit.\n"
                "\t -v,\t--version         \t, Print version information exit. \n"
//						"\t -g,\t--generator       \t, Generates a demo configure file \n"
                "\n"
                "\t--case <CASE ID>         \t, Select a case <CASE ID> to execute \n "
                "\t--case_help <CASE ID>    \t, Print a usag message of case <CASE ID> \n "


        << logger::help_message()

        << parallel::help_message()

        << io::help_message();


        MESSAGE

        << " Use case list:" << std::endl

        << "        CASE ID     | Description " << std::endl
        << "--------------------|-------------------------------------"
        << std::endl;

        for (auto const &item : applist)
        {

            MESSAGE << std::setw(19) << std::right << item.first << " |"
            << item.second->description() << std::endl;
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
            << std::endl
            << "====================================================" << std::endl
            << "   Use Case [" << item->first << "]:  " << std::endl
            << "\t" << item->second->description() << std::endl
            << "----------------------------------------------------" << std::endl;

            item->second->body(options);

            GLOBAL_COMM.barrier();
        }

        MESSAGE << "===================================================="
        << std::endl

        << "\t >>> Done <<< " << std::endl;
    }
    io::close();
    parallel::close();
    logger::close();

    return 0;

}
