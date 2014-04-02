/*
 * simpla.cpp
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "../src/io/data_stream.h"
#include "../src/simpla_defs.h"
#include "../src/utilities/log.h"
#include "../src/utilities/lua_state.h"
#include "../src/utilities/parse_command_line.h"
#include "../src/utilities/utilities.h"

#include "contexts/context.h"

using namespace simpla;

int main(int argc, char **argv)
{

	Logger::Verbose(0);

	LuaObject dict;

	size_t num_of_step = 10;

	size_t record_stride = 1;

	bool just_a_test = false;

	ParseCmdLine(argc, argv, [&](std::string const & opt,std::string const & value)->int
	{
		if(opt=="n"||opt=="num_of_step")
		{
			num_of_step =ToValue<size_t>(value);
		}
		else if(opt=="s"||opt=="record_stride")
		{
			record_stride =ToValue<size_t>(value);
		}
		else if(opt=="o"||opt=="output"||opt=="p"||opt=="prefix")
		{
			GLOBAL_DATA_STREAM.OpenFile(value);
		}
		else if(opt=="i"||opt=="input")
		{
			dict.ParseFile(value);
		}
		else if(opt=="c"|| opt=="command")
		{
			dict.ParseString(value);
		}
		else if(opt=="l"|| opt=="log")
		{
			Logger::OpenFile (value);
		}
		else if(opt=="v")
		{
			Logger::Verbose(ToValue<int>(value));
		}
		else if( opt=="verbose")
		{
			Logger::Verbose(LOG_VERBOSE);
		}
		else if(opt=="q"|| opt=="quiet")
		{
			Logger::Verbose(LOG_INFORM-1);
		}
		else if(opt=="w"|| opt=="log_width")
		{
			LoggerStreams::instance().SetLineWidth(ToValue<int>(value));
		}
		else if(opt=="g"|| opt=="generator")
		{
			INFORM
			<< ShowCopyRight() << std::endl
			<< "Too lazy to implemented it\n"<< std::endl;
			TheEnd(1);
		}
		else if(opt=="t")
		{
			just_a_test=true;
		}
		else if(opt=="V")
		{
			INFORM<<ShowShortVersion()<< std::endl;
			TheEnd(0);
		}

		else if(opt=="version")
		{
			INFORM<<ShowVersion()<< std::endl;
			TheEnd(0);
		}
		else if(opt=="help")
		{
			INFORM
			<< ShowCopyRight() << std::endl
			<< "Too lazy to write a complete help information\n"<< std::endl;
			TheEnd(0);}
		else
		{
			INFORM
			<< ShowCopyRight() << std::endl
			<<
			" -h        \t print this information\n"
			" -n<NUM>   \t number of steps\n"
			" -s<NUM>   \t recorder per <NUM> steps\n"
			" -o<STRING>\t output directory\n"
			" -i<STRING>\t configure file \n"
			" -c,--config <STRING>\t Lua script passed in as string \n"
			" -t        \t only read and parse input file, but do not process  \n"
			" -g,--generator   \t generator a demo input script file \n"
			" -v<NUM>   \t verbose  \n"
			" -V        \t print version  \n"
			" -q        \t quiet mode, standard out  \n"
			;
			TheEnd(0);
		}
		return CONTINUE;

	}

	);

	INFORM << SIMPLA_LOGO;

	LOGGER << "Parse Command Line." << DONE;

	if (!dict)
	{
		LOGGER << "Nothing to do !!";
		TheEnd(-1);
	}

	LOGGER << "Pre-Process" << START;

	GLOBAL_DATA_STREAM.OpenGroup("/Input");

	Context ctx(dict);

	if (ctx.empty())
	{
		INFORM << "illegal configure! ";
		TheEnd(-2);
	}

	else
	{
		INFORM << ctx;
	}
	// Preprocess    ====================================
	// Main Loop ============================================

	LOGGER << "\n" << SINGLELINE<< "\n";
	LOGGER << "Process " << START;

	TheStart();

	if (just_a_test)
	{
		LOGGER << "Just test configure files";
	}
	else
	{

		for (int i = 0; i < num_of_step; ++i)
		{
			LOGGER << "STEP: " << i;

			ctx.NextTimeStep();

			if (i % record_stride == 0)
			{
				ctx.Dump("/DumpData");
			}
		}
	}
	LOGGER << "Process" << DONE;

	VERBOSE << "Post-Process" << START;

	GLOBAL_DATA_STREAM.OpenGroup("/Output");

	INFORM << "OutPut Path:" << GLOBAL_DATA_STREAM.GetCurrentPath();

	VERBOSE << "Post-Process" << DONE;

	TheEnd();

}
