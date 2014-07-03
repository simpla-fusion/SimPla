/*
 * parse_config.cpp
 *
 * \date  2012-3-6
 *      \author  salmon
 */

#include "parse_config.h"
#include <boost/program_options.hpp>

Properties::Holder ParseConfig(int argc, char ** argv)
{
	Properties::Holder res(new Properties);

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

	return res;
}

