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

#include "../../src/solver/EMFluid.h"
#include "../../src/io/XDMFStream.h"

namespace simpla
{
using namespace mesh;

typedef manifold::CartesianManifold mesh_type;

// namespace simpla
int main(int argc, char **argv)
{
    using namespace simpla;

    ConfigParser options;

    try
    {
        logger::init(argc, argv);

        parallel::init(argc, argv);

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
    io::XDMFStream out_stream;

    MeshAtlas m;

    auto mesh = m.add<mesh_type>();

    auto phy_solver = m.register_solver<simpla::phy_solver::EMFluid<mesh_type>>();


    try
    {
        m.setup();

        m.check_point(out_stream);
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
        m.next_step(dt);

        if (count % check_point == 0)
            m.check_point(out_stream);

        ++count;
    }
    m.teardown();

    INFORM << "\t >>> Done <<< " << std::endl;
    MESSAGE << "====================================================" << std::endl;


    io::close();

    parallel::close();

    logger::close();

    return 0;
}

