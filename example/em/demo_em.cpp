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
#include "../../src/simulation/Context.h"
#include "../../src/manifold/pre_define/PreDefine.h"
#include "EMFluid.h"
#include "PML.h"

typedef simpla::manifold::CartesianManifold mesh_type;

using namespace simpla;

using namespace simpla::mesh;


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

        index_tuple gw{2, 2, 2};

        mesh_center->setup(options["Mesh"]).name("Center");
        mesh_center->ghost_width(gw);
        mesh_center->deploy();

        ctx.add_problem_domain<EMFluid<mesh_type >>(mesh_center->id())->setup(options).deploy();

        if (options["PML"])
        {
            typedef PML<mesh_type> pml_type;
            std::shared_ptr<mesh_type> pml_mesh[6];
            index_type w = options["PML"]["Width"].as<index_type>(5);
            index_tuple dims = mesh_center->dimensions();

            pml_mesh[0] = mesh_center->clone_as<mesh_type>("PML_0");
            pml_mesh[0]->shift(index_tuple{-w, -w, -w});
            pml_mesh[0]->stretch(index_tuple{w, dims[1] + 2 * w, dims[2] + 2 * w});
            pml_mesh[0]->deploy();
            ctx.atlas().add_block(pml_mesh[0]);
            ctx.atlas().add_adjacency_2(mesh_center, pml_mesh[0], SP_MB_SYNC);
            ctx.add_domain(std::make_shared<pml_type>(pml_mesh[0].get(), mesh_center->box()))->deploy();

            pml_mesh[1] = mesh_center->clone_as<mesh_type>("PML_1");
            pml_mesh[1]->shift(index_tuple{dims[0], -w, -w});
            pml_mesh[1]->stretch(index_tuple{w, dims[1] + 2 * w, dims[2] + 2 * w});
            pml_mesh[1]->deploy();
            ctx.atlas().add_block(pml_mesh[1]);
            ctx.atlas().add_adjacency_2(mesh_center, pml_mesh[1], SP_MB_SYNC);
            ctx.add_domain(std::make_shared<pml_type>(pml_mesh[1].get(), mesh_center->box()))->deploy();
            if (dims[1] > 1)
            {
                pml_mesh[2] = mesh_center->clone_as<mesh_type>("PML_2");
                pml_mesh[2]->shift(index_tuple{0, -w, -w});
                pml_mesh[2]->stretch(index_tuple{dims[0], w, dims[2] + 2 * w});
                pml_mesh[2]->deploy();
                ctx.atlas().add_block(pml_mesh[2]);
                ctx.atlas().add_adjacency_2(pml_mesh[2], mesh_center, SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[2], pml_mesh[0], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[2], pml_mesh[1], SP_MB_SYNC);
                ctx.add_domain(std::make_shared<pml_type>(pml_mesh[2].get(), mesh_center->box()))->deploy();


                pml_mesh[3] = mesh_center->clone_as<mesh_type>("PML_3");
                pml_mesh[3]->shift(index_tuple{0, dims[1], -w});
                pml_mesh[3]->stretch(index_tuple{dims[0], w, dims[2] + 2 * w});
                pml_mesh[3]->deploy();
                ctx.atlas().add_block(pml_mesh[3]);
                ctx.atlas().add_adjacency_2(pml_mesh[3], mesh_center, SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[3], pml_mesh[0], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[3], pml_mesh[1], SP_MB_SYNC);
                ctx.add_domain(std::make_shared<pml_type>(pml_mesh[3].get(), mesh_center->box()))->deploy();

            }
            if (dims[2] > 1)
            {
                pml_mesh[4] = mesh_center->clone_as<mesh_type>("PML_4");
                pml_mesh[4]->shift(index_tuple{0, 0, -w});
                pml_mesh[4]->stretch(index_tuple{dims[0], dims[1], w});
                pml_mesh[4]->deploy();
                ctx.atlas().add_block(pml_mesh[4]);
                ctx.atlas().add_adjacency_2(pml_mesh[4], mesh_center, SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[4], pml_mesh[0], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[4], pml_mesh[1], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[4], pml_mesh[2], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[4], pml_mesh[3], SP_MB_SYNC);
                ctx.add_domain(std::make_shared<pml_type>(pml_mesh[4].get(), mesh_center->box()))->deploy();

                pml_mesh[5] = mesh_center->clone_as<mesh_type>("PML_5");
                pml_mesh[5]->shift(index_tuple{0, 0, dims[2]});
                pml_mesh[5]->stretch(index_tuple{dims[0], dims[1], w});
                pml_mesh[5]->deploy();
                ctx.atlas().add_block(pml_mesh[5]);
                ctx.atlas().add_adjacency_2(pml_mesh[5], mesh_center, SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[5], pml_mesh[0], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[5], pml_mesh[1], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[5], pml_mesh[2], SP_MB_SYNC);
                ctx.atlas().add_adjacency_2(pml_mesh[5], pml_mesh[3], SP_MB_SYNC);
                ctx.add_domain(std::make_shared<pml_type>(pml_mesh[5].get(), mesh_center->box()))->deploy();

            }
        }
    }

    ctx.print(std::cout);

    int num_of_steps = options["number_of_steps"].as<int>(1);

    int step_of_check_points = options["step_of_check_point"].as<int>(1);

    Real stop_time = options["stop_time"].as<Real>(1);
    Real dt = options["dt"].as<Real>();

    io::cd("/start/");

    ctx.save(io::global(), 0);

    MESSAGE
    << "====================================================" <<
    std::endl;

    TheStart();

    INFORM << "\t >>> Time [" << ctx.time()

    << "] <<< " <<
    std::endl;

    Real current_time = ctx.time();
    io::cd("/checkpoint/");
    ctx.check_point(io::global());

    size_type count = 0;

    while (ctx.time() < stop_time)
    {
        ctx.run(dt);
        current_time = ctx.time();
        if (count % step_of_check_points == 0) { ctx.check_point(io::global()); }
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


