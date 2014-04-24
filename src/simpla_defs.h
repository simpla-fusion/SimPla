/*
 * simpla_defs.h
 *
 *  Created on: 2011-12-24
 *      Author: salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <string>

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif
#define SIMPLA_LOGO                                                        \
"===================================================\n"\
"|             ┏━┓╻┏┳┓┏━┓╻  ┏━┓                    |\n" \
"|             ┗━┓┃┃┃┃┣━┛┃  ┣━┫                    |\n" \
"|             ┗━┛╹╹ ╹╹  ┗━╸╹ ╹                    |\n" \
"===================================================\n"\
" SimPla, Plasma Simulator        \n"\
" Build Date: " __DATE__ " " __TIME__"                   \n"\
" ID:" IDENTIFY  "                                        \n"\
" Author: YU Zhi. All rights reserved.           \n"


inline std::string ShowShortVersion()
{
	return IDENTIFY;
}
inline std::string ShowVersion()
{
	return SIMPLA_LOGO;
}
inline std::string ShowCopyRight()
{
	return SIMPLA_LOGO;
}

/**
 *  Leave all platform dependence here
 * */

#endif /* SIMPLA_DEFS_H_ */
