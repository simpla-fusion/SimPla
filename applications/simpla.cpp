/*
 * simpla.cpp
 *
 *  Created on: 2013年11月13日
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

#include "engine/object.h"
#include "particle/particle.h"
#include "particle/pic_engine_default.h"
#include "../applications/solver/electromagnetic/pml.h"

using namespace simpla;
typedef UniformRectMesh Mesh;
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

	typedef UniformRectMesh Mesh;

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

	std::vector<Object> sp_list;

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

	std::function<void(Form<1>&, Form<2>&, Form<1> const &, Real)> field_solver;

	auto solver_type = pt.GetChild("FieldSolver").template Get<std::string>(
			"Type");

	if (solver_type == "PML")
	{
		using namespace std::placeholders;
		field_solver = std::bind(&PML<Mesh>::Eval,
				std::shared_ptr<PML<Mesh>>(
						new PML<Mesh>(mesh, pt.GetChild("FieldSolver"))), _1,
				_2, _3, _4);
	}
	else
	{
		field_solver =
				[mu0,epsilon0](Form<1>&E1, Form<2>&B1, Form<1> const & J1, Real dt)
				{
					E1 += (Curl(B1 / mu0) - J1) / epsilon0 * dt;
					B1 -= Curl(E1) * dt;
					//TODO add boundary condition
				};
	}

	Particle<Mesh, PICEngineDefault<Mesh> > ion(mesh,
			pt.GetChild("Particles").GetChild("ion"));

//	std::map<std::string, Object> particle_list;
//
	for (auto const &pt_child : pt.GetChild("Particles"))
	{
		std::string engine_type = pt_child.second.Get<std::string>("Engine");
		std::string name = pt_child.second.Get<std::string>("Name");

		std::cout << pt_child.first.as<std::string>() << std::endl;
		if (engine_type == "Default")
		{

		}
//		else if (engine_type == "GGauge8")
//		{
//			particle_list[name] = Object(
//					new Particle<Mesh, PICEngineGGauge<Mesh,8> >(mesh,
//							pt_child));
//		}
	}

	for (int i = 0; i < num_of_step; ++i)
	{
		INFORM << ">>> STEP " << i << " Start <<<";
		field_solver(E, B, J, dt);

		ion.Push(E, B);
		ion.Collect<1>(J);
		INFORM << ">>> STEP " << i << " Done <<<";
	}

	INFORM << (">>> Process DONE! <<<");
	INFORM << (">>> Post-Process DONE! <<<");
//
//// Log ============================================

}
