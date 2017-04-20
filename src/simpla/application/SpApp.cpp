//
// Created by salmon on 17-4-20.
//
#include "SpApp.h"

#include <simpla/engine/Context.h>
#include <simpla/engine/TimeIntegrator.h>
#include <simpla/parallel/all.h>
#include <simpla/utilities/Logo.h>
#include <simpla/utilities/parse_command_line.h>
#include "SpApp.h"

namespace simpla {
namespace application {

SpApp::SpApp(){};
SpApp::~SpApp() { Finalize(); };
std::shared_ptr<data::DataTable> SpApp::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    if (m_schedule_ != nullptr) {
        res->Set("Schedule", m_schedule_->Serialize());
        res->SetValue("Schedule/Type", m_schedule_->GetClassName());
    }
    return res;
}
void SpApp::Deserialize(std::shared_ptr<data::DataTable> t){

};
void SpApp::Initialize(){

};
void SpApp::SetUp() {
    m_schedule_->SetUp();
    //    MPI_Barrier(GLOBAL_COMM.comm());
    //    std::shared_ptr<data::DataTable> cfg = nullptr;
    //    std::string buffer;
    //    if (GLOBAL_COMM.rank() == 0) {
    //        m_pimpl_->m_ctx_->SetUp();
    //
    //        std::ostringstream os;
    //        os << "Context={";
    //        data::Serialize(m_pimpl_->m_ctx_->Serialize(), os, "lua");
    //        os << "}";
    //        buffer = os.str();
    //
    //        parallel::bcast_string(&buffer);
    //    } else {
    //        parallel::bcast_string(&buffer);
    //
    //        cfg = std::make_shared<data::DataTable>("lua://");
    //        cfg->backend()->Parser(buffer);
    //
    //        m_pimpl_->m_ctx_->Deserialize(cfg);
    //        m_pimpl_->m_ctx_->SetUp();
    //    }
    //    MPI_Barrier(GLOBAL_COMM.comm());
    //    VERBOSE << DOUBLELINE << std::endl;
    //    VERBOSE << "Description : " << application::SpApp::ShowDescription(app->GetName()) << std::endl;
    //    //  VERBOSE << "Application : " << *app->Serialize() << std::endl;
    //    VERBOSE << DOUBLELINE << std::endl;
};
void SpApp::Run() { m_schedule_->Run(); };
void SpApp::TearDown() { m_schedule_->TearDown(); };
void SpApp::Finalize() {
    TearDown();
    m_schedule_->Finalize();
    m_schedule_.reset();
};

std::shared_ptr<SpApp> SpApp::Create(int argc, char **argv) {
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
    return data::EnableCreateFromDataTable<SpApp>::Create(app_name);
}

}  // namespace application{
}  // namespace simpla{

using namespace simpla;

int main(int argc, char **argv) {
#ifndef NDEBUG
    logger::set_stdout_level(1000);
#endif
    parallel::init(argc, argv);

    auto app = application::SpApp::Create(argc, argv);
    app->SetUp();
    app->Run();
    app->TearDown();
    VERBOSE << DOUBLELINE << std::endl;

    parallel::close();
    logger::close();
}
