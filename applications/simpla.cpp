/* Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * simpla.cpp
 *
 *  Created on: 2011-3-1
 *      Author: salmon
 *
 *  */

#include "include/simpla_defs.h"
#include "utilities/log.h"
#include "utilities/lua_parser.h"
#include "physics/physical_constants.h"
#include "fetl/fetl.h"
#include <cmath>
#include <string>
#include <vector>
#include <list>

using namespace simpla;

int main(int argc, char **argv)
{

	Log::Verbose(0);

	//===========================================================
	//  Command Line
	namespace po = boost::program_options;

	po::options_description desc;

	desc.add_options()

	("help,h", "produce help message")

	("long_help,H", "produce long help message")

	("version,V", "display copyright and  version information")

	("verbose,v", po::value<int>()->default_value(0), "verbose level")

	("log,l", po::value<std::string>()->default_value(""), "Log file")

	("input,i", po::value<std::string>()->default_value(""),
			"Input configure file [xxx.lua]")

	("command,c", po::value<std::string>()->default_value(""),
			"command | script")

	("output,o", po::value<std::string>()->default_value("untitle"),
			"Output file, diagnose information")

	("gen_config,g", "generate example configure file")

	po::variables_map vm_;

	po::store(po::parse_command_line(argc, argv, desc), vm_);

	LuaObject pt;

	if (vm_.count("help") > 0)
	{

		std::cout << SIMPLA_LOGO << std::endl;
		std::cout << desc << std::endl;
		return (1);

	}
	else if (vm_.count("long_help") > 0)
	{

		std::cout << SIMPLA_LOGO << std::endl;
		std::cout << desc << std::endl;
		std::cout << DOUBLELINE << std::endl;
		return (1);

	}
	else if (vm_.count("gen_config") > 0)
	{
		return (1);

	}
	else if (vm_.count("version") > 0)
	{
		std::cout << SIMPLA_LOGO << std::endl;
		return (1);

	}

	Log::Verbose(vm_["verbose"].as<int>());

	Log::OpenFile(vm_["log"].as<std::string>());

	if (vm_.count("input") > 0)
	{
		pt.ParseFile(vm_["input"].as<std::string>());

	}

	if (vm_.count("command") > 0)
	{
		pt.ParseString(vm_["command"].as<std::string>());

	}

	size_t numOfStep = pt["STEP"].as<size_t>();

	PhysicalConstants phys_const;

	phys_const.Config(pt["UNIT_SYSTEM"]);

	DEFINE_FIELDS(UniformRectMesh)

	Mesh mesh;

	mesh.Config(pt["MESH"]);

	//  Parse Lua configure file ========================

	// set diagnosis fields  ====================================

//	IO::registerFunction(ctx, vm_["DIAGNOSIS"].as<std::vector<std::string> >(),
//			oFile, vm_["FORMAT"].as<std::string>(), vm_["RECORD"].as<size_t>());

	//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << SIMPLA_LOGO << std::endl;

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;

	std::cout << std::setw(20) << "Num. of procs. : " << omp_get_num_procs()
			<< std::endl;

	std::cout << std::setw(20) << "Num. of threads : " << omp_get_max_threads()
			<< std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << phys_const.Summary() << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << mesh.Summary() << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << std::endl << DOUBLELINE << std::endl;

	// Main Loop ============================================

	INFORM(">>> Pre-Process DONE! <<<");

	INFORM(">>> Process START! <<<");

	for (int i = 0; i < numOfStep; ++i)
	{
		process();
	}

	INFORM(">>> Process DONE! <<<");

	post_process();

	INFORM(">>> Post-Process DONE! <<<");

	// Log ============================================

	return (1);
}

