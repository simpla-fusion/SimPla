/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include <simpla/engine/Manager.h>
#include <simpla/parallel/all.h>
#include <simpla/toolbox/Logo.h>
#include <simpla/toolbox/parse_command_line.h>
namespace simpla {
void create_scenario(engine::Manager *ctx);
void RegisterEverything();
}
using namespace simpla;

int main(int argc, char **argv) {
#ifndef NDEBUG
    logger::set_stdout_level(1000);
#endif
    RegisterEverything();

    parallel::init(argc, argv);

    std::string output_file = "h5://SimPlaOutput";

    std::string conf_file(argv[0]);
    std::string conf_prologue = "";
    std::string conf_epilogue = "";

    conf_file += ".lua";

    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (opt == "i" || opt == "input") {
                conf_file = value;
            } else if (opt == "prologue") {
                conf_epilogue = value;
            } else if (opt == "e" || opt == "execute" || opt == "epilogue") {
                conf_epilogue = value;
            } else if (opt == "o" || opt == "output") {
                output_file = value;
            } else if (opt == "log") {
                logger::open_file(value);
            } else if (opt == "V" || opt == "verbose") {
                logger::set_stdout_level(std::atoi(value.c_str()));
            } else if (opt == "quiet") {
                logger::set_stdout_level(logger::LOG_MESSAGE - 1);
            } else if (opt == "log_width") {
                logger::set_line_width(std::atoi(value.c_str()));
            } else if (opt == "v" || opt == "version") {
                MESSAGE << "SIMPla " << ShowVersion();
                TheEnd(0);
                return TERMINATE;
            } else if (opt == "h" || opt == "help") {
                MESSAGE << ShowLogo() << std::endl
                        << " Usage: " << argv[0] << "   <options> ..." << std::endl
                        << std::endl
                        << " Options:" << std::endl
                        << std::left << std::setw(20) << "  -h, --help "
                        << ": Print a usage message and exit." << std::endl
                        << std::left << std::setw(20) << "  -v, --version "
                        << ": Print version information exit. " << std::endl
                        << std::left << std::setw(20) << "  -V, --verbose "
                        << ": Verbose mode.  Print debugging messages," << std::endl
                        << std::left << std::setw(20) << "                "
                        << "   <-10 means quiet,  >10 means Print as much as it can.(default=0)" << std::endl
                        << std::left << std::setw(20) << "  --quiet "
                        << ": quiet mode," << std::endl
                        << std::endl
                        << std::left << std::setw(20) << "  -o, --output  "
                        << ": Output file GetName (default: simpla.h5)." << std::endl
                        << std::left << std::setw(20) << "  -i, --input  "
                        << ": Input configure file (default:" + conf_file + ")" << std::endl
                        << std::left << std::setw(20) << "  -p, --prologue "
                        << ": Execute Lua script before configure file is Load" << std::endl
                        << std::left << std::setw(20) << "  -e, --epilogue "
                        << ": Execute Lua script after configure file is Load" << std::endl
                        << std::endl;

                TheEnd(0);
                return TERMINATE;
            } else {
                // options.add(opt, (value == "") ? "true" : value);
            }
            return CONTINUE;
        });

    MESSAGE << ShowLogo() << std::endl;

    MPI_Barrier(GLOBAL_COMM.comm());

    engine::Manager ctx;
    if (GLOBAL_COMM.rank() == 0) {
        create_scenario(&ctx);
        ctx.Initialize();
        std::ostringstream os;
        os << "Config=";
        data::Serialize(ctx.db(), os, "lua");
        std::string buffer = os.str();
        parallel::bcast_string(&buffer);
    } else {
        std::string buffer;
        parallel::bcast_string(&buffer);
        auto t_db = std::make_shared<data::DataTable>("lua://");
        t_db->backend()->Parser(buffer);
        ctx.db()->Set(*t_db->GetTable("Config"));
        ctx.Initialize();
    }
    MPI_Barrier(GLOBAL_COMM.comm());

    int num_of_steps = ctx.db()->GetValue<int>("number_of_steps", 1);
    int step_of_check_points = ctx.db()->GetValue<int>("step_of_check_point", 1);
    Real dt = ctx.db()->GetValue<Real>("dt", 1.0);

    MESSAGE << DOUBLELINE << std::endl;
    MESSAGE << "INFORMATION:" << std::endl;
    MESSAGE << "Context : " << *ctx.db() << std::endl;
    MESSAGE << SINGLELINE << std::endl;

    MESSAGE << DOUBLELINE << std::endl;
    TheStart();

    size_type step = 0;

    while (step <= num_of_steps) {
        ctx.Advance(dt);
        ctx.Synchronize();

        INFORM << "\t >>>  [ Time = " << ctx.GetTime() << " Step = " << step << "] <<< " << std::endl;
        if (step % step_of_check_points == 0) { data::DataTable(output_file).Set(*ctx.db()); };
        ++step;
    }
    MESSAGE << " DONE " << *ctx.db()->Get("Patches") << std::endl;
    MESSAGE << "\t >>> Done <<< " << std::endl;
    MESSAGE << DOUBLELINE << std::endl;
    TheEnd();
    parallel::close();
    logger::close();
}
