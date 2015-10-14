/*
 * @file demo_pic.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "demo_pic.h"

#include "../../core/application/application.h"
#include "../../core/application/use_case.h"

#include "utilities.h"
#include "../../core/io/io.h"
#include "../../core/physics/physical_constants.h"
#include "../../core/field/field.h"

#include "../../core/mesh/mesh.h"

#include "../../core/particle/particle.h"

using namespace simpla;

typedef StructuredMesh<geometry::coordinate_system::Cartesian<3>,
		InterpolatorLinear, FiniteDiffMethod> mesh_type;

USE_CASE(pic," Particle in cell" )
{

	size_t num_of_steps = 1000;
	size_t check_point = 10;

	if (options["case_help"])
	{

		MESSAGE

		<< " Options:" << endl

				<<

				"\t -n,\t--number_of_steps <NUMBER>  \t, "
						"Number of steps = <NUMBER> ,default="
						+ value_to_string(num_of_steps) + "\n"
								"\t -s,\t--check_point <NUMBER>          "
								"  \t, default=" + value_to_string(check_point)
						+ "\n";

		return;
	}

	options["n"].as(&num_of_steps);

	options["check_point"].as<size_t>(&check_point);

	auto mesh = std::make_shared<mesh_type>();

	mesh->load(options["Mesh"]);

	mesh->deploy();

	MESSAGE << "======== Initialize ========" << std::endl;

	auto rho = mesh->template make_form<VERTEX, Real>();
	auto J = mesh->template make_form<EDGE, Real>();
	auto E = mesh->template make_form<EDGE, Real>();
	auto B = mesh->template make_form<FACE, Real>();

	E.clear();
	B.clear();
	J.clear();

	VERBOSE_CMD(loadField(options["InitValue"]["E"], &E));
	VERBOSE_CMD(loadField(options["InitValue"]["B"], &B));

	auto J_src = makeField_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["J"]);

	auto B_src = makeField_function_by_config<FACE, Real>(*mesh,
			options["Constraint"]["B"]);

	auto E_src = makeField_function_by_config<EDGE, Real>(*mesh,
			options["Constraint"]["E"]);

	typedef PICDemo engine_type;

	auto ion = make_kinetic_particle<engine_type>(
			mesh->template domain<VERTEX>());

	size_t pic = options["Particle"]["H"]["pic"].template as<size_t>(10);

	ion->mass(options["Particle"]["H"]["mass"].template as<Real>(1.0));

	ion->charge(options["Particle"]["H"]["charge"].template as<Real>(1.0));

	ion->temperature(options["Particle"]["H"]["T"].template as<Real>(1.0));

	ion->deploy();

	auto p_generator = simple_particle_generator(*ion, mesh->local_extents(),
			ion->temperature(), options["Particle"]["H"]["Distribution"]);

	std::mt19937 rnd_gen;

	for (size_t i = 0, ie = pic * ion->domain().size(); i < ie; ++i)
	{
		ion->insert(p_generator(rnd_gen));
	}

	ion->sync();
	ion->wait();
	cd("/Input/");

	VERBOSE << save("H1", ion->dataset()) << std::endl;

	LOGGER << "----------  Show Configuration  ---------- " << endl;

	if (GLOBAL_COMM.process_num()==0)
	{

		MESSAGE << endl

		<< "[ Configuration ]" << endl

		<< " Description=\"" << options["Description"].as<std::string>("") << "\""
		<< endl

		<< " Mesh = " << endl << "  {" << *mesh << "} " << endl

		<< " Particle ="<<endl<<

		" H = {"<<*ion<<"}"<<endl

		<< " TIME_STEPS = " << num_of_steps << endl;
	}

	LOGGER << "----------  Dump input ---------- " << endl;
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
		LOG_CMD(ion->rehash());

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
