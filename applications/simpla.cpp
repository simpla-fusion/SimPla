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
#include "utilities/lua_state.h"
#include "engine/object.h"

#include "mesh/uniform_rect.h"
#include "physics/physical_constants.h"

#include "fetl/fetl.h"

#include "particle/particle.h"
#include "particle/pic_engine_default.h"

#include "solver/electromagnetic/pml.h"
using namespace simpla;

template<int IFORM> using Form = Field<Geometry<UniformRectMesh,IFORM>,Real >;

template<int IFORM> using VecForm = Field<Geometry<UniformRectMesh,IFORM>,nTuple<3,Real> >;

int main(int argc, char **argv)
{

	Log::Verbose(0);

	LuaObject pt;

	size_t num_of_step;

	size_t record_stride;

	std::string workspace_path;

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

			std::cout << SIMPLA_LOGO << std::endl;
			std::cout << "Too lazy to write a complete help information\n"
					"\t -n<NUM>\t number of steps\n"
					"\t -s<NUM>\t recorder per <NUM> steps\n"
					"\t -o<STRING>\t output directory\n"
					"\t -i<STRING>\t configure file "
					"\n" << std::endl;
			exit(1);
			break;
		default:
			std::cout << SIMPLA_LOGO << std::endl;

		}

	}

	PhysicalConstants phys_const;

	phys_const.Deserialize(pt.GetChild("UnitSystem"));

	typedef UniformRectMesh Mesh;
	Mesh mesh;

	mesh.Deserialize(pt.GetChild("Grid"));

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

	phys_const.Print(std::cout);

	std::cout << SINGLELINE << std::endl;

	mesh.Print(std::cout);

	std::cout << SINGLELINE << std::endl;

//	std::cout << std::endl << DOUBLELINE << std::endl;

// Main Loop ============================================

	const double mu0 = phys_const["permeability of free space"];
	const double epsilon0 = phys_const["permittivity of free space"];
	const double speed_of_light = phys_const["speed of light"];
	const double proton_mass = phys_const["proton mass"];
	const double elementary_charge = phys_const["elementary charge"];

	Form<1> E(mesh);
	Form<1> J(mesh);
	Form<2> B(mesh);

	Real dt = mesh.GetDt();

	std::vector<Object> sp_list;

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

//	ColdFluidEM<Mesh> cold_fluid(mesh, phys_const);

	nTuple<6, int> bc =
	{ 5, 5, 5, 5, 5, 5 };
	PML<Mesh> pml(mesh, phys_const, bc);
	Particle<PICEngineDefault<Mesh> > ion(mesh, 1.0, 1.0);
	ion.Init(100);

	for (int i = 0; i < num_of_step; ++i)
	{
		INFORM << ">>> STEP " << i << " Start <<<";
		pml.Eval(E, B, J, dt);
//		cold_fluid.Eval(E, B, J, sp_list, dt);

//		E += (Curl(B / mu0) - J) / epsilon0 * dt;
//		B -= Curl(E) * dt;

		ion.Push(E, B);
		ion.Scatter(J);
		INFORM << ">>> STEP " << i << " Done <<<";
	}

	INFORM << (">>> Process DONE! <<<");
	INFORM << (">>> Post-Process DONE! <<<");
//
//// Log ============================================

}
