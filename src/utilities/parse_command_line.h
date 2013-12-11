/*
 * parse_commandline.h
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#ifndef PARSE_COMMAND_LINE_H_
#define PARSE_COMMAND_LINE_H_

enum
{
	CONTINUE = 0, TERMINATE = 1

}
;

inline void ParseCmdLine(int argc, char **argv,
		std::function<int(std::string const &, std::string const &)> const & fun)
{
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

			if (fun(opt, value) == TERMINATE)
				break; // terminate paser stream;

			opt = "";
			value = "";
			ready_to_process = false;
		}

	}
}

#endif /* PARSE_COMMAND_LINE_H_ */
