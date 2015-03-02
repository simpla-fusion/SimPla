/*
 * pic_case.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */
#include "demo_pic.h"

#include <memory>
#include <string>
#include "../../../core/utilities/utilities.h"
#include "../../../core/io/io.h"
#include "../../../core/field/load_field.h"
#include "../../../core/field/field_constraint.h"
#include "../../../core/application/use_case.h"
#include "../../../core/diff_geometry/fetl.h"

#include "../../../core/particle/kinetic_particle.h"

using namespace simpla;

USE_CASE(em)
{

	size_t num_of_steps = 1000;
	size_t strides = 10;
	Real dt = 0.001;

	options.register_cmd_line_option<size_t>("NUMBER_OF_STEPS", "n");

	options.register_cmd_line_option<size_t>("STRIDES", "s");

	options.register_cmd_line_option<Real>("DT", "dt");

	if (options["SHOW_HELP"])
	{
		SHOW_OPTIONS("-n,--number_of_steps <NUMBER_OF_STEPS>",
				"number of steps = <NUMBER_OF_STEPS> ,default="
						+ value_to_string(num_of_steps));
		SHOW_OPTIONS("-s,--strides <STRIDES>",
				" dump record per <STRIDES> steps, default="
						+ value_to_string(strides));
		SHOW_OPTIONS("-dt  <DT>",
				" value of time step,default =" + value_to_string(dt));

		return;
	}

	options["NUMBER_OF_STEPS"].as(&num_of_steps);

	options["STRIDES"].as<size_t>(&strides);

	auto manifold = make_manifold<CartesianMesh>();

	manifold->load(options["Mesh"]);

	manifold->update();

	if (options["DT"].as<Real>(&dt))
	{
		manifold->dt(dt);
	}

	MESSAGE << std::endl;

	MESSAGE << "======== Configuration ========" << std::endl;
	MESSAGE << " Description:" << options["Description"].as<std::string>("")
			<< std::endl;
	MESSAGE << " Options:" << std::endl;
	RIGHT_COLUMN(" mesh" ) << " = {" << *manifold << "}," << std::endl;
	RIGHT_COLUMN(" time step" ) << " = " << num_of_steps << std::endl;

	MESSAGE << "======== Initlialize ========" << std::endl;
	// Load initialize value

	auto J = make_form<EDGE, Real>(manifold);
	auto E = make_form<EDGE, Real>(manifold);
	auto B = make_form<FACE, Real>(manifold);

	auto E_src = make_constraint<EDGE, Real>(manifold,
			options["Constraint"]["E"]);
	auto J_src = make_constraint<EDGE, Real>(manifold,
			options["Constraint"]["J"]);
	auto B_src = make_constraint<FACE, Real>(manifold,
			options["Constraint"]["B"]);

	VERBOSE_CMD(load(options["InitValue"]["B"], &B));
	VERBOSE_CMD(load(options["InitValue"]["E"], &E));
	VERBOSE_CMD(load(options["InitValue"]["J"], &J));

	auto ion = make_kinetic_particle<PICDemo>(manifold);

	ion->load(options["Particle"]);

	ion->properties("Cache Length") = strides;

	MESSAGE << "======== START! ========" << std::endl;

	cd("/Input/");

	VERBOSE << SAVE(E);
	VERBOSE << SAVE(B);
	VERBOSE << SAVE(J);

	cd("/Save/");

	for (size_t s = 0; s < num_of_steps; ++s)
	{

		E_src(&E);
		B_src(&B);

		J.clear();
		ion->next_timestep(dt, E, B, &J);
		J_src(&J);

		E += curl(B) * dt - J;
		B += -curl(E) * dt;

		if (s % strides == 0)
		{
			VERBOSE << save("H", *ion, SP_APPEND);
			VERBOSE << save("E", E, SP_APPEND);
			VERBOSE << save("B", B, SP_APPEND);

		}
	}
	cd("/Output/");
	VERBOSE << SAVE(E);
	VERBOSE << SAVE(B);
	VERBOSE << SAVE(J);
	VERBOSE << save("H", *ion);

	MESSAGE << "======== DONE! ========" << std::endl;

}

