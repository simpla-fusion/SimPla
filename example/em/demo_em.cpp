/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include "../../src/io/IO.h"
#include "../../src/gtl/Utilities.h"
#include "../../src/manifold/pre_define/PreDefine.h"
#include "../../src/simulation/Context.h"

#include "../../scenario/problem_domain/EMFluid.h"
#include "../../scenario/problem_domain/PML.h"

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

    simulation::Context ctx;


    ctx.print(std::cout);

    int num_of_steps = options["number_of_steps"].as<int>(1);

    int step_of_check_points = options["step_of_check_point"].as<int>(1);

    Real stop_time = options["stop_time"].as<Real>(1);
    Real dt = options["dt"].as<Real>();

    io::cd("/start/");

    ctx.save(io::global(), 0);

    MESSAGE << "====================================================" << std::endl;

    TheStart();

    INFORM << "\t >>> Time [" << ctx.time() << "] <<< " << std::endl;

    Real current_time = ctx.time();
    io::cd("/checkpoint/");
    ctx.check_point(io::global());

    size_type count = 0;

    while (ctx.time() < stop_time)
    {

        ctx.run(dt);

        current_time = ctx.time();

        if (count % step_of_check_points == 0)
        {
            ctx.check_point(io::global());
        }
        INFORM << "\t >>>  [ Time = " << current_time << " Count = " << count << "] <<< " << std::endl;

        ++count;
    }


    INFORM << "\t >>> Done <<< " << std::endl;


    // MESSAGE << "====================================================" << std::endl;
    io::cd("/dump/");
    ctx.save(io::global(), 0);
    ctx.teardown();

    TheEnd();

    io::close();
    parallel::close();
    logger::close();
}


