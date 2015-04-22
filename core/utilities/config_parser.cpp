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
std::string ConfigParser::init(int argc, char ** argv)
{
	m_lua_object_.init();

	std::string lua_file("");
	std::string lua_prologue = "";
	std::string lua_epilogue = "";

	simpla::parse_cmd_line(argc, argv,
			[&](std::string const & opt,std::string const & value)->int
			{
				if(opt=="i"||opt=="input")
				{
					lua_file=value;
				}

				else if(opt=="prologue" )
				{
					lua_epilogue=value;
				}
				else if(opt=="e"|| opt=="execute"|| opt=="epilogue")
				{
					lua_epilogue=value;
				}
				else if(opt=="i"||opt=="input")
				{
					lua_file=value;
				}
				else
				{
					if(value=="")
					{	m_kv_map_[opt]="true";}
					else
					{
						m_kv_map_[opt]=value;

					}

				}
				return CONTINUE;
			}

			);
	try
	{
		m_lua_object_.parse_string(lua_prologue);
		m_lua_object_.parse_file(lua_file);
		m_lua_object_.parse_string(lua_epilogue);

	} catch (...)
	{
		WARNING << "Can not load configure file:[" << lua_file << "]"
				<< std::endl;
	}

	return "\t -i,--input <STRING> 	\t,  input configure file \n"
			"\t --prologue <STRING>	\t,  execute Lua script before confingure file is load\n"
			"\t -e,--epilogue <STRING> \t,  execute Lua script after confingure file is load\n";

}
}  // namespace simpla
