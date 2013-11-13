/*
 * simpla.cpp
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#include <iostream>
#include <string>

#include "simpla_defs.h"
#include "mesh/uniform_rect.h"
#include "physics/physical_constants.h"
#include "utilities/log.h"
#include "utilities/lua_parser.h"
#include "fetl/fetl.h"
#include "engine/object.h"

using namespace simpla;

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
	size_t num_of_step = pt.Get<size_t>("STEP", 1000);

	PhysicalConstants phys_const;

	phys_const.Config(pt["UNIT_SYSTEM"]);

	using namespace UniformRectMeshDefine;

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

	OneForm E(mesh);
	OneForm J(mesh);
	TwoForm B(mesh);

	Real dt = mesh.GetDt();

	std::vector<CompoundObject> sp_list;

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

	for (int i = 0; i < num_of_step; ++i)
	{
		E += (Curl(B / mu0) - J) / epsilon0 * dt;

		B -= Curl(E) * dt;

		ZeroForm BB(mesh);

		VecZeroForm Ev(mesh), Bv(mesh), dEvdt(mesh);

		BB = Wedge(B, HodgeStar(B));

		VecZeroForm K_(mesh);

		VecZeroForm K(mesh);

		K.clear();

		ZeroForm a(mesh);
		ZeroForm b(mesh);
		ZeroForm c(mesh);
		a.clear();
		b.clear();
		c.clear();

		for (auto &v : sp_list)
		{
			auto & ns = v.at("n").as<ZeroForm>();
			auto & Js = v.at("J").as<VecZeroForm>();
			Real ms = v.properties.get<Real>("m") * proton_mass;
			Real Zs = v.properties.get<Real>("Z") * elementary_charge;

			ZeroForm as(mesh);

			as = 2.0 * ms / (dt * Zs);

			a += ns * Zs / as;
			b += ns * Zs / (BB + as * as);
			c += ns * Zs / ((BB + as * as) * as);

			K_ = /* 2.0 * nu * Js*/
			-2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

			K -= Js + 0.5 * (

			K_ / as

			+ Cross(K_, Bv) / (BB + as * as)

			+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as))

			);
		}

//		a = a * (0.5 * dt) / epsilon0 - 1.0;
		b = b * (0.5 * dt) / epsilon0;
		c = c * (0.5 * dt) / epsilon0;

		K /= epsilon0;

		dEvdt = K / a
				+ Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
				+ Cross(Cross(K, Bv), Bv) * (-c * c * BB + c * a - b * b)
						/ (a * ((c * BB - a) * (c * BB - a) + b * b * BB));
		for (auto &v : sp_list)
		{
			auto & ns = v.at("n").as<ZeroForm>();
			auto & Js = v.at("J").as<VecZeroForm>();
			Real ms = v.properties.get<Real>("m") * proton_mass;
			Real Zs = v.properties.get<Real>("Z") * elementary_charge;

			ZeroForm as(mesh);
			as = 2.0 * ms / (dt * Zs);

			K_ = // 2.0*nu*(Js)
					-2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;
			Js +=

			K_ / as

			+ Cross(K_, Bv) / (BB + as * as)

			+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
		}

//		J -=  dEvdt;
	}

//	INFORM << (">>> Process DONE! <<<");
//	INFORM << (">>> Post-Process DONE! <<<");
//
//// Log ============================================

}
