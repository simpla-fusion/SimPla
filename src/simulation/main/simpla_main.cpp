/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include "../../toolbox/IO.h"
#include "../../toolbox/Utilities.h"
#include "../../toolbox/Parallel.h"
#include "../Context.h"

namespace simpla
{
void create_scenario(simulation::Context *ctx, ConfigParser const &options);
}
using namespace simpla;

int main(int argc, char **argv)
{

    parallel::init(argc, argv);

    std::shared_ptr<io::IOStream> os;

    ConfigParser options;

    {
        std::string output_file = "simpla.h5";

        std::string conf_file(argv[0]);
        std::string conf_prologue = "";
        std::string conf_epilogue = "";

        conf_file += ".lua";

        simpla::parse_cmd_line(
            argc, argv,
            [&](std::string const &opt, std::string const &value) -> int
            {
                if (opt == "i" || opt == "input") { conf_file = value; }

                else if (opt == "prologue") { conf_epilogue = value; }

                else if (opt == "e" || opt == "execute" || opt == "epilogue") { conf_epilogue = value; }

                else if (opt == "o" || opt == "output") { output_file = value; }

                else if (opt == "log") { logger::open_file(value); }

                else if (opt == "V" || opt == "verbose") { logger::set_stdout_level(std::atoi(value.c_str())); }

                else if (opt == "quiet") { logger::set_stdout_level(logger::LOG_MESSAGE - 1); }

                else if (opt == "log_width") { logger::set_line_width(std::atoi(value.c_str())); }

                else if (opt == "v" || opt == "version")
                {
                    MESSAGE << "SIMPla " << ShowVersion();
                    TheEnd(0);
                    return TERMINATE;
                }

                else if (opt == "h" || opt == "help")
                {
                    /* @formatter:off */
                    MESSAGE
                        << ShowLogo() << std::endl
                        << " Usage: " << argv[0] << "   <options> ..." << std::endl
                        << std::endl
                        << " Options:" << std::endl
                        <<std::left<< std::setw(20) << "  -h, --help "     << ": Print a usage message and exit." << std::endl
                        <<std::left<< std::setw(20) << "  -v, --version "  << ": Print version information exit. " << std::endl
                        <<std::left<< std::setw(20) << "  -V, --verbose "  << ": Verbose mode.  Print debugging messages,"<<std::endl
                        <<std::left<< std::setw(20) << "                "  << "   <-10 means quiet,  >10 means print as much as it can.(default=0)"<<std::endl
                        <<std::left<< std::setw(20) << "  --quiet "        << ": quiet mode,"<<std::endl
                        <<std::endl
                        <<std::left<< std::setw(20) << "  -o, --output  "  << ": Output file name (default: simpla.h5)." << std::endl
                        <<std::left<< std::setw(20) << "  -i, --input  "   << ": Input configure file (default:" + conf_file + ")" << std::endl
                        <<std::left<< std::setw(20) << "  -p, --prologue " << ": Execute Lua script before configure file is load" << std::endl
                        <<std::left<< std::setw(20) << "  -e, --epilogue " << ": Execute Lua script after configure file is load" << std::endl
                        <<std::endl;

                    /* @formatter:on*/
                    TheEnd(0);
                    return TERMINATE;
                }
                else { options.add(opt, (value == "") ? "true" : value); }

                return CONTINUE;

            }


        );


        MESSAGE << ShowLogo() << std::endl;

        options.parse(conf_file, conf_prologue, conf_epilogue);

        os = io::create_from_output_url(output_file);
    }

    simulation::Context ctx;

    create_scenario(&ctx, options);

    MESSAGE << DOUBLELINE << std::endl;
    MESSAGE << "INFORMATION:                                        " << std::endl;
    MESSAGE << "Context = {" << ctx << " }" << std::endl;
    MESSAGE << SINGLELINE << std::endl;

    int num_of_steps = options["number_of_steps"].as<int>(1);

    int step_of_check_points = options["step_of_check_point"].as<int>(1);

    Real dt = options["dt"].as<Real>();

    os->open("/start/");

    ctx.save(*os);

    MESSAGE << DOUBLELINE << std::endl;

    TheStart();

    INFORM << "\t >>> Time [" << ctx.time() << "] <<< " << std::endl;

    os->open("/checkpoint/");

    ctx.sync();

    ctx.check_point(*os);

    size_type count = 0;

    while (count <= num_of_steps)
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

    MESSAGE << DOUBLELINE << std::endl;

    TheEnd();
    os->close();
    parallel::close();
    logger::close();

}
