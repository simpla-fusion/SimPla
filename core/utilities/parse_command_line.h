/*
 * parse_commandline.h
 *
 *  created on: 2013-12-11
 *      Author: salmon
 */

#ifndef PARSE_COMMAND_LINE_H_
#define PARSE_COMMAND_LINE_H_

#include <functional>
#include <string>

namespace simpla
{

enum
{
	CONTINUE = 0, TERMINATE = 1
}
;
/**
 * \ingroup Configure
 * \brief Command line parser
 *
 *
 * example:
 * @code{.Cpp}
 * 	ParseCmdLine(argc, argv,
 *		[&](std::string const & opt,std::string const & value)->int
 *		{
 *			if(opt=="n"||opt=="num_of_step")
 *			{
 *				num_of_step =ToValue<size_t>(value);
 *			}
 *			else if(opt=="s"||opt=="record_stride")
 *			{
 *				record_stride =ToValue<size_t>(value);
 *			}
 *			else if(opt=="i"||opt=="input")
 *			{
 *				dict.ParseFile(value);
 *			}
 *			else if(opt=="c"|| opt=="command")
 *			{
 *				dict.ParseString(value);
 *			}
 *			else if(opt=="version")
 *			{
 *				INFORM<<ShowVersion()<< std::endl;
 *				TheEnd(0);
 *			}
 *			else if(opt=="help")
 *			{
 *				INFORM
 *				<< ShowCopyRight() << std::endl
 *				<< "Too lazy to write a complete help information\n"<< std::endl;
 *				TheEnd(0);
 *			}
 *
 *			return CONTINUE;
 *		}
 *
 *);
 * @endcode
 *
 * @param argc
 * @param argv
 * @param options  response operation to configure options ,  std::function<int(std::string const &, std::string const &)>
 *
 */
inline void ParseCmdLine(int argc, char **argv,
        std::function<int(std::string const &, std::string const &)> const & options)
{
	int i = 1;

	bool ready_to_process = false;
	std::string opt = "";
	std::string value = "";

	while (i < argc)
	{
		char * str = argv[i];
		if (str[0] == '-' && ((str[1] < '0' || str[1] > '9') && (str[1] != '.'))) // is configure flag
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

}  // namespace simpla
#endif /* PARSE_COMMAND_LINE_H_ */
