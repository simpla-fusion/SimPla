/*
 * @file demo_pic.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "demo_pic.h"


#include "../../src/gtl/Utilities.h"
#include "../../src/io/IO.h"
#include "../../src/physics/PhysicalConstants.h"

#include "../../src/manifold/pre_define/riemannian.h"

#include "../../src/particle/ParticleContainer.h"

#include "../../src/field/field.h"
#include "../../src/particle/obsolete/SimpleParticleGenerator.h"

//#include "../../src/field/field_dense.h"
//#include "../../src/field/field_traits.h"
//#include "../../src/field/field_function.h"

using namespace simpla;

typedef manifold::Riemannian<3> mesh_type;

USE_CASE(pic, " particle in cell")
{

    size_t num_of_steps = 1000;
    size_t check_point = 10;

    if (options["case_help"]) {

        /*	MESSAGE

                    << " Options:" << endl

                    <<

                    "\t -n,\t--number_of_steps <NUMBER>  \t, "
                            "Number of steps = <NUMBER> ,default="
                    //		+ value_to_string(num_of_steps) + "\n"
                        //	"\t -s,\t--check_point <NUMBER>          "
                        //	"  \t, default=" + value_to_string(check_point)
                            + "\n";
    */
        return;
    }

    options["n"].as(&num_of_steps);

    options["check_point"].as<size_t>(&check_point);

    auto mesh = std::make_shared<mesh_type>();

    mesh->load(options);

    mesh->deploy();

    MESSAGE << "======== Initialize ========" << std::endl;

    auto rho = traits::make_field<VERTEX, Real>(*mesh);
    auto J = traits::make_field<EDGE, Real>(*mesh);
    auto E = traits::make_field<EDGE, Real>(*mesh);
    auto B = traits::make_field<FACE, Real>(*mesh);

    E.clear();
    B.clear();
    J.clear();

//	VERBOSE_CMD(load_field(options["InitValue"]["E"], &E));
//	VERBOSE_CMD(load_field(options["InitValue"]["B"], &B));

    auto J_src = traits::make_function_by_config<Real>(
            options["Constraint"]["J"], traits::make_domain<EDGE>(*mesh));

    auto B_src = traits::make_function_by_config<Real>(
            options["Constraint"]["B"], traits::make_domain<FACE>(*mesh));

    auto E_src = traits::make_function_by_config<Real>(
            options["Constraint"]["E"], traits::make_domain<EDGE>(*mesh));


    auto ion = std::make_shared<Particle<pic_demo, mesh_type>>(*mesh);

    size_t pic = options["particle"]["H"]["pic"].template as<size_t>(10);

    ion->mass(options["particle"]["H"]["mass"].template as<Real>(1.0));

    ion->charge(options["particle"]["H"]["charge"].template as<Real>(1.0));

    ion->temperature(options["particle"]["H"]["T"].template as<Real>(1.0));

    ion->deploy();

    auto p_generator = simple_particle_generator(*ion, mesh->extents(),
                                                 ion->temperature(), options["particle"]["H"]["Distribution"]);

    std::mt19937 rnd_gen;

    for (size_t i = 0, ie = pic; i < ie; ++i) {
        ion->insert(p_generator(rnd_gen));
    }

    ion->sync();
    ion->wait();
    cd("/Input/");

    VERBOSE << save("H1", ion->dataset()) << std::endl;

    LOGGER << "----------  Show Configuration  ---------- " << std::endl;

    if (GLOBAL_COMM.process_num() == 0) {

        MESSAGE << std::endl

        << "[ Configuration ]" << std::endl

        << " Description=\"" << options["Description"].as<std::string>("") << "\""
        << std::endl

        << " mesh = " << std::endl << "  {" << *mesh << "} " << std::endl

        << " particle =" << std::endl

        //			    <<	" H = {" << *ion << "}" << std::endl

        << " TIME_STEPS = " << num_of_steps << std::endl;
    }

    LOGGER << "----------  Dump input ---------- " << std::endl;
    VERBOSE << SAVE(E) << std::endl;
    VERBOSE << SAVE(B) << std::endl;

    DEFINE_PHYSICAL_CONST
    Real dt = mesh->dt();
    auto dx = mesh->dx();

    Real omega = 0.01 * PI / dt;

    LOGGER << "----------  START ---------- " << std::endl;

    cd("/Save/");

    for (size_t step = 0; step < num_of_steps; ++step) {
        VERBOSE << "Step [" << step << "/" << num_of_steps << "]" << std::endl;

        J.clear();

        LOG_CMD(ion->next_time_step(dt, E, B, &J));
        LOG_CMD(ion->rehash());

        J = (J_src);
        B = (B_src);

        LOG_CMD(E += curl(B) * (dt * speed_of_light2) - J * (dt / epsilon0));
        LOG_CMD(B -= curl(E) * dt);

        E = (E_src);

        VERBOSE << SAVE_RECORD(J) << std::endl;
        VERBOSE << SAVE_RECORD(E) << std::endl;
        VERBOSE << SAVE_RECORD(B) << std::endl;

        mesh->next_time_step();

    }
    MESSAGE << "======== DONE! ========" << std::endl;

    cd("/Output/");

    VERBOSE << save("H", ion->dataset()) << std::endl;
    VERBOSE << SAVE(E) << std::endl;
    VERBOSE << SAVE(B) << std::endl;
    VERBOSE << SAVE(J) << std::endl;
}
