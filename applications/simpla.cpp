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
#include "mesh/uniform_rect.h"
#include "particle/particle.h"
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <boost/program_options.hpp>
#include <omp.h>

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

	;

	po::variables_map vm_;

	po::store(po::parse_command_line(argc, argv, desc), vm_);

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

	LuaObject pt;

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

	const double mu0 = phys_const["permeability_of_free_space"];
	const double epsilon0 = phys_const["permittivity_of_free_space"];
	const double speed_of_light = phys_const["speed_of_light"];
	const double proton_mass = phys_const["proton_mass"];
	const double elementary_charge = phys_const["elementary_charge"];

	OneForm E(mesh);
	OneForm J(mesh);
	TwoForm B(mesh);

	Real dt = mesh.dt;

	std::map<std::string, Particle<Mesh> > sp_list;

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

	for (int i = 0; i < numOfStep; ++i)
	{
		E += (Curl(B / mu0) - J) / epsilon0 * dt;

		B -= Curl(E) * dt;

		ZeroForm BB(mesh);

		VecZeroForm Ev(mesh), Bv(mesh), dEvdt(mesh);

		BB = Dot(Bv, Bv);

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
			ZeroForm & ns = v.second.n;
			VecZeroForm & Js = v.second.J;
			Real ms = v.second.m * proton_mass;
			Real Zs = v.second.Z * elementary_charge;
			ZeroForm as;
			as.fill(2.0 * ms / (dt * Zs));

			a += ns * Zs / as;
			b += ns * Zs / (BB + as * as);
			c += ns * Zs / ((BB + as * as) * as);

			K_ = // 2.0*nu*Js
					-2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

			K -= Js
					+ 0.5
							* (K_ / as + Cross(K_, Bv) / (BB + as * as)
									+ Cross(Cross(K_, Bv), Bv)
											/ (as * (BB + as * as)));

		}

		a = a * (0.5 * dt) / epsilon0 - 1.0;
		b = b * (0.5 * dt) / epsilon0;
		c = c * (0.5 * dt) / epsilon0;

		K /= epsilon0;

		dEvdt = K / a
				+ Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
				+ Cross(Cross(K, Bv), Bv) * (-c * c * BB + c * a - b * b)
						/ (a * ((c * BB - a) * (c * BB - a) + b * b * BB));
		for (auto &v : sp_list)

		{
			ZeroForm & ns = v.second.n;
			VecZeroForm & Js = v.second.J;

			Real ms = v.second.m * proton_mass;
			Real Zs = v.second.Z * elementary_charge;

			ZeroForm as;
			as.fill(2.0 * ms / (dt * Zs));

			K_ = // 2.0*nu*(Js)
					-2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;
			Js += K_ / as + Cross(K_, Bv) / (BB + as * as)
					+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
		}

		//	J -= MapTo(Int2Type<IOneForm>(), dEvdt);
	}

	INFORM << (">>> Process DONE! <<<");
	INFORM << (">>> Post-Process DONE! <<<");

// Log ============================================

	return (1);
}

