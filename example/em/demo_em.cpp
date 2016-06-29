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
#include "../../src/parallel/Parallel.h"
#include "../../src/simulation/Context.h"

#include "EMFluid.h"
#include "PML.h"

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

    {
        auto mesh_center = ctx.add_mesh<mesh_type>();

        {
            auto c_mesh = ctx.get_mesh<mesh_type>(mesh_center);
            c_mesh->name("Center");
            c_mesh->setup(options["Mesh"]);
            c_mesh->deploy();
        }

        ctx.add_problem_domain<EMFluid<mesh_type>>(mesh_center)
                ->setup(options).deploy();

        if (options["PML"])
        {
            size_type PML_width = 5;
            auto &atlas = ctx.get_mesh_atlas();

            int od[3];
            int count = 0;
            for (int tag = 1, tag_e = 1 << 6; tag < tag_e; tag <<= 1)
            {

                od[0] = ((tag & 0x3) << 1) - 3;
                od[1] = (((tag >> 2) & 0x3) << 1) - 3;
                od[2] = (((tag >> 4) & 0x3) << 1) - 3;

                if (od[0] > 1 || od[1] > 1 || od[2] > 1)
                {
                    continue;
                }


                auto b_id = atlas.extent_block(mesh_center, od, PML_width);

                ctx.get_mesh_block(b_id)->name("PML_" + type_cast<std::string>(count));

                ctx.add_problem_domain<PML<mesh_type>>(b_id)
                        ->set_direction(od).deploy();


                ++count;
            }
        }
//
//
//    {
//        std::string str = options["ProblemDomain"].as<std::string>();
//        if (str == "PIC")
//        {
//            problem_domain = std::make_shared<EMPIC < mesh_type>>
//            (&mesh_center);
//        }
//        else if (str == "Fluid")
//        {
//            problem_domain = std::make_shared<EMFluid<mesh_type>>(&mesh_center);
//
//        }
//        else
//        {
//            RUNTIME_ERROR << "Unknown problem type [" << str << "]" << std::endl;
//        }
//
//    }


    }
    ctx.print(std::cout);


    Real stop_time = options["stop_time"].as<Real>(ctx.time());

    int num_of_steps = options["number_of_steps"].as<int>(1);

    Real inc_time = (stop_time - ctx.time()) /
                    (options["number_of_check_point"].as<int>(1));
    io::cd("/start/");
    ctx.save(io::global());

    MESSAGE << "====================================================" << std::endl;

    TheStart();

    INFORM << "\t >>> Time [" << ctx.time() << "] <<< " << std::endl;

    Real current_time = ctx.time();
    io::cd("/checkpoint/");
    ctx.check_point(io::global());

    while (ctx.time() < stop_time)
    {

        ctx.run(current_time + inc_time);

        current_time = ctx.time();

        ctx.check_point(io::global());

        INFORM << "\t >>> Time [" << current_time << "] <<< " << std::endl;

    }


    INFORM << "\t >>> Done <<< " << std::endl;


    // MESSAGE << "====================================================" << std::endl;
    io::cd("/dump/");
    ctx.save(io::global());
    ctx.teardown();

    TheEnd();

    io::close();
    parallel::close();
    logger::close();
}


