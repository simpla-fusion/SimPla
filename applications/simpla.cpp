/* Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * simpla.cpp
 *
 *  Created on: 2011-3-1
 *      Author: salmon
 *
 *  */

#include "defs.h"
#include <boost/program_options.hpp>
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include "engine/engine.h"
#include "io/io.h"

int main(int argc, char **argv)
{

	logger.setLevel(0);

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
			"Input configure file [xxx.lua.cfg]")

	("output,o", po::value<std::string>()->default_value("untitle"),
			"Output file, diagnose information")

	("FORMAT", po::value<std::string>()->default_value("HDF5"),
			"The format of output file, HDF5 or XDMF ")

	("gen_config,g", "generate example configure file")

	("RECORD,r", po::value<size_t>()->default_value(1),
			"Interval between two record")

	("STEP,s", po::value<size_t>()->default_value(10), "Number of time step.")

	("DIAGNOSIS,d", po::value<std::vector<std::string> >()->multitoken(),
			"Fields need to be diagnosed");

	po::variables_map vm_;

	po::store(po::parse_command_line(argc, argv, desc), vm_);

	std::string oFile = vm_["output"].as<std::string>();

	std::string cFile("");

	if (vm_.count("help") > 0)
	{

		std::cout << Context::Copyright << std::endl;
		std::cout << desc << std::endl;
		return (1);

	}
	else if (vm_.count("long_help") > 0)
	{

		std::cout << Context::Copyright << std::endl;
		std::cout << desc << std::endl;
		std::cout << DOUBLELINE << std::endl;
		std::cout << Context::ExampleConfigFile << std::endl;
		std::cout << DOUBLELINE << std::endl;
		return (1);

	}
	else if (vm_.count("gen_config") > 0)
	{
		std::cout << Context::ExampleConfigFile << std::endl;
		return (1);

	}
	else if (vm_.count("version") > 0)
	{
		std::cout << Context::Copyright << std::endl;
		return (1);

	}
	else
	{

		logger.setLevel(vm_["verbose"].as<int>());

		logger.setLogFile(vm_["log"].as<std::string>());

		cFile = vm_["input"].as<std::string>();
	}

	size_t numOfStep = vm_["STEP"].as<size_t>();

	//  Parse Lua configure file ========================

	Context::Holder ctx(parseConfigFile(cFile));

	// set diagnosis fields  ====================================

	IO::registerFunction(ctx, vm_["DIAGNOSIS"].as<std::vector<std::string> >(),
			oFile, vm_["FORMAT"].as<std::string>(),
			vm_["RECORD"].as<size_t>());

	//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << Context::Copyright << std::endl;

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;

	std::cout << std::setw(20) << "Num. of procs. : " << omp_get_num_procs()
			<< std::endl;

	std::cout << std::setw(20) << "Num. of threads : " << omp_get_max_threads()
			<< std::endl;

	std::cout << std::setw(20) << "Configure File : " << cFile << std::endl;

	std::cout << std::setw(20) << "Output Path: " << oFile << std::endl;

	std::cout << std::setw(20) << "Number of steps : " << numOfStep
			<< std::endl;

	std::cout << std::setw(20) << "Record/steps : "
			<< vm_["RECORD"].as<size_t>() << std::endl;

	std::cout << std::setw(20) << "Full Time : " << ctx->grid->dt * numOfStep
			<< "[s]" << std::endl;

	std::cout << SINGLELINE << std::endl;

	ctx->showSummary();

	std::cout << std::endl << DOUBLELINE << std::endl;

	// Main Loop ============================================

	ctx->pre_process();

	INFORM(">>> Pre-Process DONE! <<<");

	INFORM(">>> Process START! <<<");

	for (int i = 0; i < numOfStep; ++i)
	{
		ctx->process();
	}

	INFORM(">>> Process DONE! <<<");

	ctx->post_process();

	INFORM(">>> Post-Process DONE! <<<");

	// Log ============================================

	logger.summary();

	return (1);
}

