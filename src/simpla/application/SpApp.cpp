/**
 * @file SpApp.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include "SpApp.h"
#include <simpla/engine/Context.h>
#include <simpla/engine/TimeIntegrator.h>
#include <simpla/parallel/all.h>
#include <simpla/utilities/Logo.h>
#include <simpla/utilities/parse_command_line.h>

using namespace simpla;

namespace simpla {
namespace application {
SpApp::SpApp() {}
SpApp::~SpApp() {}
std::shared_ptr<data::DataTable> SpApp::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    if (m_schedule_ != nullptr) {
        res->Set("Schedule", m_schedule_->Serialize());
        res->SetValue("Schedule/Type", m_schedule_->GetClassName());
    }
    return res;
};
void SpApp::Deserialize(std::shared_ptr<data::DataTable> t) {
    m_schedule_ = engine::Schedule::Create(t->GetTable("Schedule"));
};

void SpApp::Initialize(){};
void SpApp::Finalize(){};
void SpApp::SetUp() {
    if (m_schedule_ != nullptr) { m_schedule_->SetUp(); }
};
void SpApp::TearDown() {
    if (m_schedule_ != nullptr) {
        m_schedule_->TearDown();
        m_schedule_.reset();
    }
};
void SpApp::Run() {
    if (m_schedule_ != nullptr) { m_schedule_->Run(); }
};
void SpApp::SetSchedule(std::shared_ptr<engine::Schedule> s) { m_schedule_ = s; }
std::shared_ptr<engine::Schedule> SpApp::GetSchedule() const { return m_schedule_; }

}  // namespace application{
}  // namespace simpla{

int main(int argc, char **argv) {
#ifndef NDEBUG
    logger::set_stdout_level(1000);
#endif

    parallel::init(argc, argv);

    std::string output_file = "h5://SimPlaOutput";

    std::string conf_file(argv[0]);
    std::string conf_prologue = "";
    std::string conf_epilogue = "";
    std::string app_name = "";
    conf_file += ".lua";

    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (opt == "i" || opt == "input") {
                conf_file = value;
            } else if (opt == "case") {
                app_name = value;
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
                        << ": Verbose mode.  Print debugging messages,   <-10 means quiet,  >10 means Print as much as "
                           "it can.(default=0)"
                        << std::endl
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
                        << std::left << std::setw(20) << "  --app "
                        << ": application name" << std::endl
                        << std::endl
                        << std::endl
                        << std::endl;
                MESSAGE << application::SpApp::ShowDescription() << std::endl;
                TheEnd(0);
                return TERMINATE;
            } else {
                // options.add(opt, (value == "") ? "true" : value);
            }
            return CONTINUE;
        });
    MESSAGE << ShowLogo() << std::endl;
    //    if (!application::SpApp::HasCreator(app_name)) {
    //        MESSAGE << application::SpApp::ShowDescription() << std::endl;
    //        RUNTIME_ERROR << "Can not find App Creator:" << app_name << std::endl;
    //    }

    MPI_Barrier(GLOBAL_COMM.comm());
    std::shared_ptr<application::SpApp> app = nullptr;
    std::shared_ptr<data::DataTable> cfg = nullptr;
    std::string buffer;
    if (GLOBAL_COMM.rank() == 0) {
        //        cfg = data::ParseCommandLine(argc, argv);
        //        app->Deserialize(cfg);
        app = application::SpApp::Create(app_name);
        app->SetUp();

        std::ostringstream os;
        os << "Application={";
        data::Serialize(app->Serialize(), os, "lua");
        os << "}";
        buffer = os.str();
        parallel::bcast_string(&buffer);
    } else {
        parallel::bcast_string(&buffer);
        cfg = std::make_shared<data::DataTable>("lua://");
        cfg->backend()->Parser(buffer);
        app = application::SpApp::Create(cfg->GetValue<std::string>("Application"));
        ASSERT(app != nullptr);
        app->SetUp();
    }
    MPI_Barrier(GLOBAL_COMM.comm());

    VERBOSE << DOUBLELINE << std::endl;
    VERBOSE << "Description : " << application::SpApp::ShowDescription(app_name) << std::endl;
    //  VERBOSE << "Application : " << *app->Serialize() << std::endl;
    VERBOSE << DOUBLELINE << std::endl;

    app->Initialize();

    TheStart();
    MPI_Barrier(GLOBAL_COMM.comm());
    app->Run();
    MPI_Barrier(GLOBAL_COMM.comm());
    TheEnd();
    app->Finalize();

    VERBOSE << DOUBLELINE << std::endl;

    parallel::close();
    logger::close();
}
