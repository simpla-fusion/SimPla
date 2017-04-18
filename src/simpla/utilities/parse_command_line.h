/**
 * @file  parse_command_line.h
 *
 *  created on: 2013-12-11
 *      Author: salmon
 */

#ifndef PARSE_COMMAND_LINE_H_
#define PARSE_COMMAND_LINE_H_

#include <cstdbool>
#include <functional>
#include <string>
#include <tuple>

namespace simpla
{
/**
 * @ingroup toolbox
 * @{
 * @addtogroup configuration Configuration Parser
 * }
 */
enum
{
    CONTINUE = 0, TERMINATE = 1
};

/**
 * @ingroup  configuration
 * @brief Command line parser
 *
 *
 * example:
 * @code{.cpp}
 * 	ParseCmdLine(argc, argv,
 *		[&](std::string const & opt,std::string const & value)->int
 *		{
 *			if(opt=="n"||opt=="num_of_step")
 *			{
 *				num_of_step =string_to_value<size_t>(value);
 *			}
 *			else if(opt=="s"||opt=="record_stride")
 *			{
 *				record_stride =string_to_value<size_t>(value);
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
 *				<< ShowLogo() << std::endl
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
 * @param fun  response operation to configure options ,  std::function<int(std::string const &, std::string const &)>
 *
 */
void parse_cmd_line(int argc, char **argv,
                    std::function<int(std::string const &, std::string const &)> const &fun);

/**
 * @ingroup  configuration
 * @brief find an options from command line
 * @param argc
 * @param argv
 * @param key
 * @return  if key is found return {true, option string} else return {false,...}
 */
std::tuple<bool, std::string> find_option_from_cmd_line(int argc, char **argv,
                                                        std::string const &key);

/**@}*/
}  // namespace simpla
#endif /* PARSE_COMMAND_LINE_H_ */
