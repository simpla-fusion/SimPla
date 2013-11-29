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
#include "mesh/co_rect_mesh.h"

#include "utilities/log.h"
#include "utilities/lua_state.h"

#include "engine/object.h"
#include "particle/particle.h"
#include "particle/pic_engine_default.h"

#include "../applications/solver/electromagnetic/pml.h"
#include "../applications/pic/pic_gauge.h"
#include "../applications/pic/pic_delta_f.h"

using namespace simpla;

void help_mesage()
{
	std::cout << "Too lazy to write a complete help information\n"
			"\t -n<NUM>\t number of steps\n"
			"\t -s<NUM>\t recorder per <NUM> steps\n"
			"\t -o<STRING>\t output directory\n"
			"\t -i<STRING>\t configure file "
			"\n" << std::endl;
}

DEFINE_FIELDS(CoRectMesh)

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

	INFORM << (">>> Pre-Process Start! <<<");

	std::function<void(Form<1>&, Form<2>&, Form<1> const &, Real)> field_solver;

	auto solver_type = pt.GetChild("FieldSolver").template Get<std::string>(
			"Type");

	if (solver_type == "PML")
	{
		using namespace std::placeholders;
		auto *solver = new PML<Mesh>(mesh);
		solver->Deserialize(pt.GetChild("FieldSolver"));
		field_solver = std::bind(&PML<Mesh>::Eval,
				std::shared_ptr<PML<Mesh>>(solver), _1, _2, _3, _4);
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

	std::map<std::string, std::shared_ptr<ParticleBase<Mesh> >> particle_list;

	for (auto const &pt_child : pt.GetChild("Particles"))
	{
		std::string engine_type = pt_child.second.Get<std::string>("Engine");
		std::string name = pt_child.second.Get<std::string>("Name");

		std::shared_ptr<ParticleBase<Mesh> > point;

		if (engine_type == "Default")
		{
			std::shared_ptr<Particle<Mesh, PICEngineDefault<Mesh>> > p(
					new Particle<Mesh, PICEngineDefault<Mesh>>(mesh));

			p->Deserialize(pt_child.second);

			point = std::dynamic_pointer_cast<ParticleBase<Mesh> >(p);

		}
		else if (engine_type == "Deltaf")
		{
			std::shared_ptr<Particle<Mesh, PICEngineDeltaF<Mesh>> > p(
					new Particle<Mesh, PICEngineDeltaF<Mesh>>(mesh));

			p->Deserialize(pt_child.second);

			point = std::dynamic_pointer_cast<ParticleBase<Mesh> >(p);

		}
		else if (engine_type == "GGauge8")
		{
			std::shared_ptr<Particle<Mesh, PICEngineGGauge<Mesh, 8>> > p(
					new Particle<Mesh, PICEngineGGauge<Mesh, 8> >(mesh));
			p->Deserialize(pt_child.second);

			point = std::dynamic_pointer_cast<ParticleBase<Mesh> >(p);
		}
		else if (engine_type == "GGauge32")
		{
			std::shared_ptr<Particle<Mesh, PICEngineGGauge<Mesh, 32>> > p(
					new Particle<Mesh, PICEngineGGauge<Mesh, 32> >(mesh));
			p->Deserialize(pt_child.second);

			point = std::dynamic_pointer_cast<ParticleBase<Mesh> >(p);
		}

		particle_list.insert(std::make_pair(name, point));

	}

	INFORM << (">>> Pre-Process DONE! <<<");
	INFORM << (">>> Process START! <<<");

	for (int i = 0; i < num_of_step; ++i)
	{
		INFORM << ">>> STEP " << i << " Start <<<";

		field_solver(E, B, J, dt);

		for (auto & p : particle_list)
		{
			INFORM << "Push Particle " << p.first;
			p.second->Push(E, B);
			INFORM << "Collect Current J from Particle " << p.first;
			p.second->Collect<1>(J, E, B);
		}

		INFORM << ">>> STEP " << i << " Done <<<";
	}

	INFORM << (">>> Process DONE! <<<");
	INFORM << (">>> Post-Process DONE! <<<");
//
//// Log ============================================

}
