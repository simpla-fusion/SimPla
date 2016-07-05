/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include "../../io/IO.h"
#include "../../gtl/Utilities.h"
#include "../../parallel/Parallel.h"
#include "../Context.h"

namespace simpla
{
void create_scenario(simulation::Context *ctx, ConfigParser const &options);
}
using namespace simpla;

int main(int argc, char **argv)
{

    ConfigParser options;
    logger::init(argc, argv);
#ifndef NDEBUG
    logger::set_stdout_level(20);
#endif

    parallel::init(argc, argv);
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
    std::shared_ptr<io::IOStream> os = io::create_from_args(argc, argv);

    simulation::Context ctx;

    create_scenario(&ctx, options);

    ctx.print(std::cout);

    int num_of_steps = options["number_of_steps"].as<int>(1);
    int step_of_check_points = options["step_of_check_point"].as<int>(1);
    Real stop_time = options["stop_time"].as<Real>(1);
    Real dt = options["dt"].as<Real>();

    os->open("/start/");

    ctx.save(*os);

    MESSAGE << "====================================================" << std::endl;

    TheStart();

    INFORM << "\t >>> Time [" << ctx.time() << "] <<< " << std::endl;

    os->open("/checkpoint/");

    ctx.sync();
    ctx.check_point(*os);

    size_type count = 0;
    while (ctx.time() <= stop_time)
    {

        ctx.run(dt);

        ctx.sync();

        if (count % step_of_check_points == 0) { ctx.check_point(*os); }

        INFORM << "\t >>>  [ Time = " << ctx.time() << " Count = " << count << "] <<< " << std::endl;

        ++count;
    }
    INFORM << "\t >>> Done <<< " << std::endl;


    os->open("/dump/");
    ctx.save(*os);
    ctx.teardown();
    MESSAGE << "====================================================" << std::endl;
    TheEnd();
    os->close();
    parallel::close();
    logger::close();

}
