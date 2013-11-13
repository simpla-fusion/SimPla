/*
 * simpla.cpp
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#include <iostream>
#include <string>

#include "simpla_defs.h"
#include "utilities/log.h"
#include "utilities/lua_parser.h"
#include "engine/object.h"

#include "mesh/uniform_rect.h"
#include "physics/physical_constants.h"

#include "fetl/fetl.h"

#include "particle/particle.h"
#include "particle/pic_engine_default.h"
using namespace simpla;

template<int IFORM> using Form = Field<Geometry<UniformRectMesh,IFORM>,Real >;

template<int IFORM> using VecForm = Field<Geometry<UniformRectMesh,IFORM>,nTuple<3,Real> >;

int main(int argc, char **argv)
{

	Log::Verbose(0);
//
////	//===========================================================
////	//  Command Line
//////	namespace po = boost::program_options;
//////
//////	po::options_description desc;
//////
//////	desc.add_options()
//////
//////	("help,h", "produce help message")
//////
//////	("long_help,H", "produce long help message")
//////
//////	("version,V", "display copyright and  version information")
//////
//////	("verbose,v", po::value<int>()->default_value(0), "verbose level")
//////
//////	("log,l", po::value<std::string>()->default_value(""), "Log file")
//////
//////	("input,i", po::value<std::string>()->default_value(""),
//////			"Input configure file [xxx.lua]")
//////
//////	("command,c", po::value<std::string>()->default_value(""),
//////			"command | script")
//////
//////	("output,o", po::value<std::string>()->default_value("untitle"),
//////			"Output file, diagnose information")
//////
//////	("gen_config,g", "generate example configure file")
//////
//////	;
//////
//////	po::variables_map vm_;
//////
//////	po::store(po::parse_command_line(argc, argv, desc), vm_);
//////
//////	if (vm_.count("help") > 0)
//////	{
//////
//////		std::cout << SIMPLA_LOGO << std::endl;
//////		std::cout << desc << std::endl;
//////		return (1);
//////
//////	}
//////	else if (vm_.count("long_help") > 0)
//////	{
//////
//////		std::cout << SIMPLA_LOGO << std::endl;
//////		std::cout << desc << std::endl;
//////		std::cout << DOUBLELINE << std::endl;
//////		return (1);
//////
//////	}
//////	else if (vm_.count("gen_config") > 0)
//////	{
//////		return (1);
//////
//////	}
//////	else if (vm_.count("version") > 0)
//////	{
//////		std::cout << SIMPLA_LOGO << std::endl;
//////		return (1);
//////
//////	}
////
//////	Log::Verbose(vm_["verbose"].as<int>());
//////
//////	Log::OpenFile(vm_["log"].as<std::string>());
//////
	LuaObject pt;
//
//	for (int i = 1; i < argc; ++i)
//	{
//		switch (argv[i][1])
//		{
//		case 'n':
//			max_step = atoi(argv[i] + 2);
//			break;
//		case 's':
//			record_stride = atoi(argv[i] + 2);
//			break;
//		case 'o':
//			output = argv[i] + 2;
//			break;
//		case 'i':
//			input = argv[i] + 2;
//			break;
//		case 'l':
//			log_file = std::string(argv[i] + 2);
//			break;
//		case 'v':
//			Log::Verbose(atof(argv[i] + 2));
//			break;
//		}
//
//	}
//

	typedef UniformRectMesh Mesh;

	size_t num_of_step = pt.get<size_t>("STEP", 1000);

	PhysicalConstants phys_const;

	phys_const.Config(pt["UNIT_SYSTEM"]);

	Mesh mesh;

	mesh.Config(pt["MESH"]);

	//  Parse Lua configure file ========================

	//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << SIMPLA_LOGO << std::endl;

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;
////
//////	std::cout << std::setw(20) << "Num. of procs. : " << omp_get_num_procs()
//////			<< std::endl;
//////
//////	std::cout << std::setw(20) << "Num. of threads : " << omp_get_max_threads()
//////			<< std::endl;
//
	std::cout << SINGLELINE << std::endl;

	std::cout << phys_const.Summary() << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << mesh.Summary() << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << std::endl << DOUBLELINE << std::endl;

// Main Loop ============================================

	const double mu0 = phys_const["permeability_of_free_space"];
	const double epsilon0 = phys_const["permittivity_of_free_space"];
	const double speed_of_light = phys_const["speed_of_light"];
	const double proton_mass = phys_const["proton_mass"];
	const double elementary_charge = phys_const["elementary_charge"];

	Form<1> E(mesh);
	Form<1> J(mesh);
	Form<2> B(mesh);

	Real dt = mesh.get_dt();

	std::vector<CompoundObject> sp_list;

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

//	ColdFluidEM<Mesh> cold_fluid(mesh, phys_const);

	Particle<PICEngineDefault<Mesh> > ion(mesh, 1.0, 1.0);

	for (int i = 0; i < num_of_step; ++i)
	{
//		cold_fluid.Eval(E, B, J, sp_list, dt);

		E += (Curl(B / mu0) - J) / epsilon0 * dt;
		B -= Curl(E) * dt;
		ion.Push(E, B);
		ion.Scatter(J);
	}

//	INFORM << (">>> Process DONE! <<<");
//	INFORM << (">>> Post-Process DONE! <<<");
//
//// Log ============================================

}
