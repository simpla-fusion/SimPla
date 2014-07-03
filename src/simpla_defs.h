/**
 * \file simpla_defs.h
 *
 *    \date 2011-12-24
 *    \author: salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <string>
#include "utilities/log.h"

/**
 *
 *
 * \mainpage Design of SimPla
 *
 * \b SimPla is a unified and hierarchical development framework for plasma simulation.
 * Its long term goal is to provide complete modeling of a fusion device.
 * “SimPla” is abbreviation of four words,  __Simulation__, __Integration__, __Multi-physics__ and __Plasma__.
 *
 * \section Contents Contents
 *
 * \li \ref Background
 * \li \ref Detail
 * \li \ref SeeAlso
 *
 * \section  Background  Background
 *  In the tokamak, from edge to core, the physical processes of plasma have different temporal-spatial
 *  scales and are described by different physical models. To achieve the device scale simulation, these
 *  physical models should be integrated into one simulation system. A reasonable solution is to reuse
 *  and couple existing software, i.e. integrated modeling projects IMI, IMFIT and TRANSP. However,
 *  different codes have different data structures and different interfaces, which make it a big challenge
 *  to efficiently integrate them together. Therefore, we consider another more aggressive solution,
 *  implementing and coupling different physical models and numerical algorithms on a unified framework
 *  with sharable data structures and software architecture. This is maybe more challenging, but can solve
 *  the problem by the roots.
 *  There are several important advantages to implement a unified software framework for the tokamak
 *  simulation system.
 *  - Different physical models could be tightly and efficiently coupled together. Data are shared in
 *    memory, and inter-process communications are minimized.
 *  - Decoupling and reusing physics independent functions, the implementation of new physical theory
 *    and model would be much easier.
 *  - Decoupling and reusing physics independent functions, the performance could be optimized by
 *    non-physicists, without any affection on the physical validity. Physicist   can easily take the
 *     benefit from the rapid growth of HPC.
 *  - All physical models and numerical algorithms applied into the simulation system could be comprehensively
 *    reviewed.
 *  To completely recover the physical process in the tokamak device, we need create a simulation system
 *   consisting of several different physical models. A unified development framework is necessary to
 *   achieve this goal.
 *
 * \section  Detail  Detail
 *\f[
 *   |I_2|=\left| \int_{0}^T \psi(t)
 *            \left\{
 *               u(a,t)-
 *               \int_{\gamma(t)}^a
 *               \frac{d\theta}{k(\theta,t)}
 *               \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
 *            \right\} dt
 *         \right|
 * \f]
 *
 *
 * \section   SeeAlso See Also
 *
 *
 * Your more detailed description here.
 *
 * \li \ref Logging
 *
 * \li \ref Versioning
 *
 * \page   Versioning API Versioning
 *
 * Overview of API Versioning
 *
 *
 * \link   Versioning View All Versioning Classes \endlink
 *
 * \defgroup   Versioning Versioning System
 * See \ref Versioning for a detailed description.
 *
 */

namespace simpla
{

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif
#define SIMPLA_LOGO      "\n"                                    \
"===================================================\n"\
"|             ┏━┓╻┏┳┓┏━┓╻  ┏━┓                    |\n" \
"|             ┗━┓┃┃┃┃┣━┛┃  ┣━┫                    |\n" \
"|             ┗━┛╹╹ ╹╹  ┗━╸╹ ╹                    |\n" \
"===================================================\n"\
" SimPla, Plasma Simulator        \n"\
" Build Date: " __DATE__ " " __TIME__"                   \n"\
" ID:" IDENTIFY  "                                        \n"\
" \author  YU Zhi. All rights reserved.           \n"

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

inline void TheStart(int flag = 1)
{
	switch (flag)
	{
	default:
		INFORM << SINGLELINE;
		VERBOSE << "So far so good, let's start work! ";
		INFORM << "[MISSOIN     START]: " << TimeStamp;
		INFORM << SINGLELINE;
	}
}
inline void TheEnd(int flag = 1)
{
	switch (flag)
	{
	case -2:
		INFORM << "Oop! Some thing wrong! Don't worry, maybe not your fault!\n"
				" Just maybe! Please Check your configure file again! ";
		break;
	case -1:
		INFORM << "Sorry! I can't help you now! Please, Try again later!";
		break;
	case 0:
		INFORM << "See you! ";
		break;
	case 1:
	default:
		LOGGER << "MISSION COMPLETED!";

		INFORM << SINGLELINE;
		INFORM << "[MISSION COMPLETED]: " << TimeStamp;
		VERBOSE << "Job is Done!! ";
		VERBOSE << "	I'm so GOOD!";
		VERBOSE << "		Thanks me please!";
		VERBOSE << "			Thanks me please!";
		VERBOSE << "You are welcome!";
		INFORM << SINGLELINE;

	}
	LOGGER << std::endl;
	INFORM << std::endl;

	exit(0);
}

/**
 *  Leave all platform dependence here
 * */

}
 // namespace simpla
#endif /* SIMPLA_DEFS_H_ */
