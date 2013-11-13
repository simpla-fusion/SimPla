/*
 * ggauge.cpp
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#include <boost/optional/optional.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../src/fetl/primitives.h"
#include "../src/utilities/log.h"
#include "../src/utilities/properties.h"

using namespace simpla;

int main(int argc, char **argv)
{

	size_t max_step = 1000;

	size_t record_stride = 1;

	std::string input = "simpla.xml";
	std::string output = "untitled";
	std::string log_file = "simpla.log";

	for (int i = 1; i < argc; ++i)
	{
		switch (argv[i][1])
		{
		case 'n':
			max_step = atoi(argv[i] + 2);
			break;
		case 's':
			record_stride = atoi(argv[i] + 2);
			break;
		case 'o':
			output = argv[i] + 2;
			break;
		case 'i':
			input = argv[i] + 2;
			break;
		case 'l':
			log_file = std::string(argv[i] + 2);
			break;
		case 'v':
			Log::Verbose(atof(argv[i] + 2));
			break;
		}

	}

	Log::OpenFile(output + "/" + log_file);

	PTree pt;

	read_file(input, pt);

	boost::optional<std::string> ot = pt.get_optional<std::string>(
			"Topology.<xmlattr>.Type");
	if (!ot || *ot != "CoRectMesh")
	{
		ERROR << "Grid type mismatch";
	}

	PHYS_CONSTANTS.SetBaseUnits(
			pt.get("Context.PhysConstants.<xmlattr>.Type", "NATURE"),
			pt.get("Context.PhysConstants.m", 1.0d),
			pt.get("Context.PhysConstants.s", 1.0d),
			pt.get("Context.PhysConstants.kg", 1.0d),
			pt.get("Context.PhysConstants.C", 1.0f),
			pt.get("Context.PhysConstants.K", 1.0f),
			pt.get("Context.PhysConstants.mol", 1.0d));

	grid.SetGeometry(

	pt.get("Context.Grid.Time.<xmlattr>.dt", 1.0d),

	pt.get<Vec3>("Context.Grid.Geometry.XMin"),

	pt.get<Vec3>("Context.Grid.Geometry.XMax"),

	pt.get<IVec3>("Context.Grid.Topology.<xmlattr>.Dimensions"),

	pt.get<IVec3>("Context.Grid.Topology.<xmlattr>.Ghostwidth")

	);

//	INFORM
//
//	<< std::endl
//
//	<< DOUBLELINE << std::endl
//
//	<< SIMPLA_LOGO << std::endl
//
//	<< DOUBLELINE << std::endl
//
//	<< std::setw(20) << "Teimstamp : " << Log::Teimstamp() << std::endl
//
//	<< std::setw(20) << "Num. of procs. : " << omp_get_num_procs() << std::endl
//
//	<< std::setw(20) << "Num. of threads : " << omp_get_max_threads()
//
//	<< std::endl
//
//	<< std::setw(20) << "Configure File : " << input << std::endl
//
//	<< std::setw(20) << "Output Path : "
//
//	<< output << std::endl
//
//	<< std::setw(20) << "Log File : " << log_file << std::endl
//
//	<< std::setw(20) << "Number of steps : "
//
//	<< max_step << std::endl
//
//	<< std::setw(20) << "Record/steps : "
//
//	<< record_stride << std::endl
//
//	<< SINGLELINE << std::endl;
//
//	INFORM
//
//	<< Summary(ctx) << std::endl
//
//	<< SINGLELINE << std::endl
//
//	<< std::endl
//
//	;
//
//	INFORM << "====== Preprocess! =======" << std::endl;
//
//	ctx.InitLoad(pt.get_child("Context.InitLoad"));
//
//	INFORM << "====== Process! =======" << std::endl;
//
//	ctx.Process(pt.get_child("Context.Process"));
//
//	INFORM << "====== Done! =======" << std::endl;

}

