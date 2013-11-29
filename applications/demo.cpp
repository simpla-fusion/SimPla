/*
 * demo.cpp
 *
 *  Created on: 2013年11月23日
 *      Author: salmon
 */

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "simpla_defs.h"

#include "fetl/fetl.h"
#include "mesh/uniform_rect.h"

#include "utilities/log.h"
#include "utilities/lua_state.h"


using namespace simpla;

typedef CoRectMesh Mesh;
template<int IFORM> using Form = Field<Geometry<Mesh,IFORM>,Real >;
template<int IFORM> using VecForm = Field<Geometry<Mesh,IFORM>,nTuple<3,Real> >;

void help_mesage()
{
	std::cout << "Too lazy to write a complete help information\n"
			"\t -n<NUM>\t number of steps\n"
			"\t -s<NUM>\t recorder per <NUM> steps\n"
			"\t -o<STRING>\t output directory\n"
			"\t -i<STRING>\t configure file "
			"\n" << std::endl;
}
int main(int argc, char **argv)
{

	std::cout << SIMPLA_LOGO << std::endl;

	Log::Verbose(0);

	LuaObject pt;

	size_t num_of_step;

	size_t record_stride;

	std::string workspace_path;

	if (argc <= 1)
	{
		help_mesage();
		exit(1);
	}

	for (int i = 1; i < argc; ++i)
	{
		char opt = *(argv[i] + 1);
		char * value = argv[i] + 2;

		switch (opt)
		{
		case 'n':
			num_of_step = atoi(value);
			break;
		case 's':
			record_stride = atoi(value);
			break;
		case 'o':
			workspace_path = value;
			break;
		case 'i':
			pt.ParseFile(value);
			break;
		case 'l':
			Log::OpenFile(value);
			break;
		case 'v':
			Log::Verbose(atof(value));
			break;
		case 'h':
			help_mesage();
			exit(1);
			break;
		default:
			std::cout << SIMPLA_LOGO << std::endl;

		}

	}

	typedef CoRectMesh Mesh;

	Mesh mesh;

	mesh.Deserialize(pt.GetChild("Grid"));

//  Summary    ====================================

	std::cout << std::endl << DOUBLELINE << std::endl;

	std::cout << "[Main Control]" << std::endl;

	std::cout << SINGLELINE << std::endl;

	mesh.Print(std::cout);

	std::cout << SINGLELINE << std::endl;

// Main Loop ============================================

	const double mu0 = mesh.phys_constants["permeability of free space"];
	const double epsilon0 = mesh.phys_constants["permittivity of free space"];
	const double speed_of_light = mesh.phys_constants["speed of light"];
	const double proton_mass = mesh.phys_constants["proton mass"];
	const double elementary_charge = mesh.phys_constants["elementary charge"];

	Form<1> E(mesh);
	Form<1> J(mesh);
	Form<2> B(mesh);

	Real dt = mesh.GetDt();

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

	Particle<Mesh, PICEngineDefault<Mesh> > ion(mesh,
			pt.GetChild("Particles").GetChild("ion"));

	for (int i = 0; i < num_of_step; ++i)
	{
		INFORM << ">>> STEP [" << i << " ]<<<";

		B -= Curl(E) * (dt * 0.5);
		E += (Curl(B / mu0) - J) / epsilon0 * dt;
		B -= Curl(E) * (dt * 0.5);

		ion.Push(E, B);
		ion.Collcet(J);
	}

	INFORM << (">>> Process DONE! <<<");
	INFORM << (">>> Post-Process DONE! <<<");
//
//// Log ============================================

}
