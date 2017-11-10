//
// Created by salmon on 17-9-4.
//
#include <simpla/geometry/GeoEngine.h>
#include <simpla/utilities/Logo.h>
#include <simpla/utilities/parse_command_line.h>
#include "simpla/parallel/MPIComm.h"
#include "simpla/parallel/Parallel.h"
#include "simpla/utilities/Log.h"
namespace simpla {
int Initialize(int argc, char **argv) {
#ifndef NDEBUG
    logger::set_stdout_level(1000);
#endif

    std::string geo_engine;
    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (false) {
            } else if (opt == "v") {
                logger::set_stdout_level(static_cast<size_type>(std::atoi(value.c_str())));
            } else if (opt == "geo_engine") {
                geo_engine = value;
            }
            return CONTINUE;
        });

    parallel::Initialize(argc, argv);
    geometry::Initialize(geo_engine);

    GLOBAL_COMM.barrier();
    MESSAGE << std::endl << ShowLogo() << std::endl;
    GLOBAL_COMM.barrier();
    return SP_SUCCESS;
}
int Finalize() {
    geometry::Finalize();
    GLOBAL_COMM.barrier();
    MESSAGE << std::endl << "========  DONE ========" << std::endl;
    GLOBAL_COMM.barrier();
    parallel::Finalize();
    return SP_SUCCESS;
}
}  // namespace simpla