/**
 * @file  parse_command_line.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "parse_command_line.h"
namespace simpla
{
void parse_cmd_line(int argc, char **argv,
		std::function<int(std::string const &, std::string const &)> const & options)
{
	if (argc <= 1 || argv == nullptr)
	{
		return;
	}

	std::string opt = "";
	std::string value = "";

	for (int i = 0; i < argc; ++i)
	{
		char * str = argv[i];

		if (str[0] == '-'
				&& ((str[1] < '0' || str[1] > '9') && (str[1] != '.'))) // is configure flag
		{
			if (opt != "" || value != "")
			{
				if (options(opt, value) == TERMINATE)
				{
					return;
				}
				opt = "";
				value = "";
			}

			if (str[1] == '-') // is long configure flag
			{
				opt = str + 2;
			}
			else // is short configure flag
			{
				opt = str[1];
				if (str[2] != '\0')
				{
					value = str + 2;
				}
			}

		}
		else
		{
			value += " ";
			value += str;
		}
	}

	if (opt != "" || value != "")
	{
		options(opt, value);
	}

}

std::tuple<bool, std::string> find_option_from_cmd_line(int argc, char ** argv,
		std::string const & key)
{
	std::string res("");
	bool is_found = false;

	parse_cmd_line(argc, argv,

	[&](std::string const & opt,std::string const & value)->int
	{
		if(opt=="V")
		{
			res=value;
			is_found=true;
			return TERMINATE;
		}
		else
		{
			return CONTINUE;
		}
	}

	);
	return std::make_tuple(is_found, res);
}

}  // namespace simpla
