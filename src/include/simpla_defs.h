/*
 * simpla_defs.h
 *
 *  Created on: 2011-12-24
 *      Author: salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <cstddef>
#include <cassert>
#include <limits>
#include <complex>
#include <stdint.h>
#include <cassert>

//#include "mpl/typetraits.h"

#include <memory>
#include <functional>

#include "utilities/log.h"

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif
#define SIMPLA_LOGO                                                        \
"┏━┓╻┏┳┓┏━┓╻  ┏━┓ \n" \
"┗━┓┃┃┃┃┣━┛┃  ┣━┫ \n" \
"┗━┛╹╹ ╹╹  ┗━╸╹ ╹ \n" \
" \n"\
" SimPla,  Simulating Plasma       \n"\
" Build Date: " __DATE__ " " __TIME__"                   \n"\
" ID:" IDENTIFY  "                                        \n"\
" Copyright (C) 2007-2012 YU Zhi. All rights reserved.           \n"

#ifdef NDEBUG
#  define ASSERT(_EXP_)
#else
#  define ASSERT(_COND_)    assert(_COND_);
#endif

#endif /* SIMPLA_DEFS_H_ */
