/**
 * @file config_parser.cpp
 *
 *  Created on: 2014年11月24日
 *      Author: salmon
 */

#include "config_parser.h"
#include "log.h"
#include "parse_command_line.h"

namespace simpla
{
ConfigParser::ConfigParser()
{

}

ConfigParser::~ConfigParser()
{
}
void ConfigParser::init(int argc, char ** argv)
{
	dict_type::init();

	if(argc<=1)
	{
		dict_type::parse_string("SHOW_HELP=true");
	}

	argc_ = argc;
	argv_ = argv;
	simpla::parse_cmd_line(argc, argv,
			[&](std::string const & opt,std::string const & value)->int
			{
				if(opt=="i"||opt=="input")
				{
					dict_type::parse_file(value);
				}
				else if(opt=="e"|| opt=="execute")
				{
					dict_type::parse_string(value);
				}
				else if (opt=="t"|| opt=="test")
				{
					dict_type::parse_string("JUST_A_TEST=true");
					// FIXME LuaObjet set global variable error!
//					dict_type::set("JUST_A_TEST",true);
					return TERMINATE;
				}
				else if (opt=="h"|| opt=="help")
				{
					SHOW_OPTIONS("-t,--test","only test configure file");
					SHOW_OPTIONS("-i,--input <STRING>","input configure file");
					SHOW_OPTIONS("-e,--execute <STRING>","execute Lua script as configuration");

					dict_type::parse_string("JUST_A_TEST=true");
					dict_type::parse_string("SHOW_HELP=true");
//					dict_type::set("SHOW_HELP",true);
//					dict_type::set("JUST_A_TEST",true);

					return TERMINATE;
				}
				return CONTINUE;

			}

			);

}
}  // namespace simpla
