/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */


#include "../../src/gtl/Utilities.h"
#include "../../src/parallel/Parallel.h"
#include "../../src/io/IO.h"
#include "../../src/manifold/pre_define/PreDefine.h"
#include "../../src/particle/pre_define/BorisParticle.h"
#include "EMFluid.h"

using namespace simpla;

using namespace simpla::mesh;

typedef simpla::manifold::CartesianManifold mesh_type;

int main(int argc, char **argv)
{
    using namespace simpla;

    ConfigParser options;


    logger::init(argc, argv);
#ifndef NDEBUG
    logger::set_stdout_level(20);
#endif
    parallel::init(argc, argv);

    io::init(argc, argv);

    options.init(argc, argv);


    INFORM << ShowCopyRight() << std::endl;

    if (options["V"] || options["version"])
    {
        MESSAGE << "SIMPla " << ShowVersion();
        TheEnd(0);
        return TERMINATE;
    }
    else if (options["h"] || options["help"])
    {

        MESSAGE << " Usage: " << argv[0] << "   <options> ..." << std::endl << std::endl;

        MESSAGE << " Options:" << std::endl

        << "\t -h,\t--help            \t, Print a usage message and exit.\n"

        << "\t -v,\t--version         \t, Print version information exit. \n"

        << std::endl;


        TheEnd(0);

    }


    mesh_type mesh;

    mesh.setup(options["Mesh"]);

    auto problem_domain = std::make_shared<EMFluid<mesh_type>>(&mesh);

    problem_domain->setup(options);

    problem_domain->print(std::cout);


    Real stop_time = options["stop_time"].as<Real>(problem_domain->time() + problem_domain->dt());

    int num_of_steps = options["number_of_steps"].as<int>(1);


    Real inc_time = (stop_time - problem_domain->time()) /
                    (options["number_of_check_point"].as<int>(1));


    MESSAGE << "====================================================" << std::endl;
    INFORM << "\t >>> START <<< " << std::endl;
    INFORM << "\t >>> Time [" << problem_domain->time() << "] <<< " << std::endl;

    Real current_time = problem_domain->time();
    io::cd("/checkpoint/");
    problem_domain->check_point(io::global());

    while (problem_domain->time() < stop_time)
    {

        problem_domain->run(current_time + inc_time);

        current_time = problem_domain->time();

        problem_domain->check_point(io::global());

        INFORM << "\t >>> Time [" << current_time << "] <<< " << std::endl;

    }


    INFORM << "\t >>> Done <<< " << std::endl;


    MESSAGE << "====================================================" << std::endl;
    io::cd("/dump/");

    problem_domain->save(io::global());

    problem_domain->teardown();

    problem_domain.reset();

    io::close();

    parallel::close();

    logger::close();

}


