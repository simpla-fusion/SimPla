/*
 * @file demo_pic.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "demo_pic.h"

#include "../../core/application/application.h"
#include "../../core/application/use_case.h"

#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"
#include "../../core/physics/physical_constants.h"
#include "../../core/field/field.h"

#include "../../core/mesh/mesh.h"

#include "../../core/particle/particle.h"

using namespace simpla;
typedef CartesianRectMesh mesh_type;

USE_CASE(pic," Particle in cell" )
{

	size_t num_of_steps = 1000;
	size_t strides = 10;

	if (options["case_help"])
	{

		MESSAGE

		<< " Options:" << endl

				<<

				"\t -n,\t--number_of_steps <NUMBER>  \t, Number of steps = <NUMBER> ,default="
						+ value_to_string(num_of_steps)
						+ "\n"
								"\t -s,\t--strides <NUMBER>            \t, Dump record per <NUMBER> steps, default="
						+ value_to_string(strides) + "\n";

		return;
	}

	options["n"].as(&num_of_steps);

	options["s"].as<size_t>(&strides);

	auto mesh = std::make_shared<mesh_type>();

	mesh->dimensions(options["dimensions"].template as(nTuple<size_t, 3>( { 10,
			10, 10 })));

	mesh->extents(options["xmin"].template as(nTuple<Real, 3>( { 0, 0, 0 })),
			options["xmax"].template as(nTuple<Real, 3>( { 1, 1, 1 })));

	mesh->dt(options["dt"].as<Real>(1.0));

	mesh->deploy();

	if (GLOBAL_COMM.process_num()==0)
	{

		MESSAGE << endl

		<< "[ Configuration ]" << endl

		<< " Description=\"" << options["Description"].as<std::string>("") << "\""
		<< endl

		<< " Mesh =" << endl << "  {" << *mesh << "} " << endl

		<< " TIME_STEPS = " << num_of_steps << endl;
	}

	MESSAGE << "======== Initialize ========" << std::endl;

	auto J = mesh->template make_form<EDGE, Real>();
	auto E = mesh->template make_form<EDGE, Real>();
	auto B = mesh->template make_form<FACE, Real>();

	VERBOSE_CMD(load_field(options["InitValue"]["E"], &E));
	VERBOSE_CMD(load_field(options["InitValue"]["B"], &B));

	auto J_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["J"]);

	auto B_src = make_field_function_by_config<FACE, Real>(*mesh,
			options["Constraint"]["B"]);

	auto E_src = make_field_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["E"]);

	typedef PICDemo engine_type;

	size_t pic = 10;

	options["pic"].as(&pic);

	auto ion = make_kinetic_particle<engine_type>(
			mesh->template domain<VOLUME>());

	ion->mass(1.0);

	ion->charge(2.0);

	ion->temperature(3.0);

	ion->deploy();

	auto extents = mesh->extents();

	auto p_generator = simple_particle_generator(*ion, extents, 1.0);

	std::mt19937 rnd_gen;

	for (int i = 0, ie = 1000; i < ie; ++i)
	{
		ion->insert(p_generator(rnd_gen));
	}

	LOGGER << "----------  Dump input ---------- " << endl;

	cd("/Input/");

	VERBOSE << save("H1", ion->dataset()) << std::endl;

	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;

	DEFINE_PHYSICAL_CONST
	Real dt = mesh->dt();
	auto dx = mesh->dx();

	Real omega = 0.01 * PI / dt;

	LOGGER << "----------  START ---------- " << endl;

	cd("/Save/");

	for (size_t step = 0; step < num_of_steps; ++step)
	{
		VERBOSE << "Step [" << step << "/" << num_of_steps << "]" << endl;

		J.clear();

		LOG_CMD(ion->next_timestep(dt, E, B, &J));

		J.self_assign(J_src);
		B.self_assign(B_src);

		LOG_CMD(E += curl(B) * (dt * speed_of_light2) - J * (dt / epsilon0));
		LOG_CMD(B -= curl(E) * dt);

		E.self_assign(E_src);

		VERBOSE << SAVE_RECORD(J) << endl;
		VERBOSE << SAVE_RECORD(E) << endl;
		VERBOSE << SAVE_RECORD(B) << endl;

		mesh->next_timestep();

	}
	MESSAGE << "======== DONE! ========" << std::endl;

	cd("/Output/");

	VERBOSE << save("H", ion->dataset()) << std::endl;
	VERBOSE << SAVE(E) << endl;
	VERBOSE << SAVE(B) << endl;
	VERBOSE << SAVE(J) << endl;
}
