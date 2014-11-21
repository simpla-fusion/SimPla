/*
 * parse_command_line.cpp
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

	int i = 1;

	bool ready_to_process = false;
	std::string opt = "";
	std::string value = "";

	while (i < argc)
	{
		char * str = argv[i];
		if (str[0] == '-'
				&& ((str[1] < '0' || str[1] > '9') && (str[1] != '.'))) // is configure flag
		{
			if (opt == "") // if buffer is not empty, clear it
			{

				if (str[1] == '-') // is long configure flag
				{
					opt = str + 2;
					++i;
				}
				else // is short configure flag
				{
					opt = str[1];
					if (str[2] != '\0')
					{
						value = str + 2;
					}
					++i;
				}
			}
			else
			{
				ready_to_process = true;
			}

		}
		else
		{
			value = str;
			++i;
			ready_to_process = true;
		}

		if (ready_to_process || i >= argc)  // buffer is ready to process
		{

			if (options(opt, value) == TERMINATE)
				break; // terminate paser stream;

			opt = "";
			value = "";
			ready_to_process = false;
		}

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
