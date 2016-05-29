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

#include "../../src/task_flow/Context.h"
#include "EMFluid.h"

using namespace simpla;

using namespace simpla::mesh;

typedef simpla::manifold::CartesianManifold mesh_type;

int main(int argc, char **argv)
{
    using namespace simpla;

    ConfigParser options;

    try
    {
        logger::init(argc, argv);

//        parallel::init(argc, argv);

        options.init(argc, argv);
    }
    catch (std::exception const &error)
    {
        RUNTIME_ERROR << "Initial error" << error.what() << std::endl;
    }

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

    std::shared_ptr<io::IOStream> out_stream;


    task_flow::Context ctx;

    auto mesh = ctx.m.add<mesh_type>();

    auto phy_solver = ctx.register_solver<EMFluid<mesh_type>>();


    try
    {
        ctx.setup();

        ctx.check_point(*out_stream);
    }
    catch (std::exception const &error)
    {
        RUNTIME_ERROR << "Context initialize error!" << error.what() << std::endl;
    }


    size_t count = 0;

    int dt = options["dt"].as<Real>(20);

    int num_of_steps = options["number_of_steps"].as<int>(20);

    int check_point = options["check_point"].as<int>(1);


    MESSAGE << "====================================================" << std::endl;
    INFORM << "\t >>> START <<< " << std::endl;

    while (count < num_of_steps)
    {
        INFORM << "\t >>> STEP [" << count << "] <<< " << std::endl;

        ctx.next_step(dt);

        phy_solver->next_step(dt);

        if (count % check_point == 0)
            ctx.check_point(*out_stream);

        ++count;
    }
    ctx.teardown();

    INFORM << "\t >>> Done <<< " << std::endl;
    MESSAGE << "====================================================" << std::endl;


//    io::close();

//    parallel::close();

    logger::close();

}


