/*
 * simpla_defs.h
 *
 *  Created on: 2011-12-24
 *      Author: salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <memory>

//#include <stdint.h>
//#include "utilities/log.h"
//#include <cassert>
//#include <complex>
//#include <cstddef>
//#include <functional>
//#include <limits>
//#include <cstddef>

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif
#define SIMPLA_LOGO                                                        \
"===================================================\n"\
"|             ┏━┓╻┏┳┓┏━┓╻  ┏━┓                    |\n" \
"|             ┗━┓┃┃┃┃┣━┛┃  ┣━┫                    |\n" \
"|             ┗━┛╹╹ ╹╹  ┗━╸╹ ╹                    |\n" \
"===================================================\n"\
" SimPla, Plasma Simulation        \n"\
" Build Date: " __DATE__ " " __TIME__"                   \n"\
" ID:" IDENTIFY  "                                        \n"\
" Copyright (C) 2007-2012 YU Zhi. All rights reserved.           \n"

#ifdef NDEBUG
#  define ASSERT(_EXP_)
#else
#  define ASSERT(_COND_)    assert(_COND_);
#endif


/**
 *  Leave all platform dependence here
 * */

#endif /* SIMPLA_DEFS_H_ */
