/**
 * @file use_case_main.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */
#include <iostream>
#include <string>

#include "use_case.h"
#include "../io/IO.h"
#include "Parallel.h"
#include "../sp_config.h"
#include "../gtl/utilities/log.h"
#include "../gtl/utilities/config_parser.h"
#include "../gtl/utilities/logo.h"


std::shared_ptr<simpla::use_case::UseCase> u_case;

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


    options.init(argc, argv);

    INFORM << ShowCopyRight() << std::endl;

    if (options["V"] || options["version"])
    {
        MESSAGE << "SIMPla " << ShowVersion();
        TheEnd(0);
        return TERMINATE;
    }
    else if (options["h"] || options["help"])
    {

        MESSAGE << " Usage: " << argv[0] << "   <options> ..." << std::endl << std::endl;

        MESSAGE

        << " Options:" << std::endl

        <<

        "\t -h,\t--help            \t, Print a usage message and exit.\n"
                "\t -v,\t--version         \t, Print version information exit. \n"
//						"\t -g,\t--generator       \t, Generates a demo configure file \n"
                "\n"
                "\t--case <CASE ID>         \t, Select a case <CASE ID> to execute \n "
                "\t--case_help <CASE ID>    \t, Print a usag message of case <CASE ID> \n ";

        MESSAGE

        << " Use case list:" << std::endl

        << "        CASE ID     | Description " << std::endl
        << "--------------------|-------------------------------------"
        << std::endl;

//        for (auto const &item : case_list)
//        {
//
//            MESSAGE << std::setw(19) << std::right << item.first << " |"
//            << item.second->description() << std::endl;
//        }

        TheEnd(0);

    }
//    MESSAGE
//    << std::endl
//    << "====================================================" << std::endl
//    << "   Use Case [" << u_case->first << "]:  " << std::endl
//    << "\t" << u_case->second->description() << std::endl
//    << "----------------------------------------------------" << std::endl;
//
    u_case->setup(argc, argv);

    u_case->next_time_step(1.0);
//
//    u_case->teardown();
//
//    MESSAGE << "===================================================="
//    << std::endl
//
//    << "\t >>> Done <<< " << std::endl;

    io::close();
    parallel::close();
    logger::close();

    return 0;

}
