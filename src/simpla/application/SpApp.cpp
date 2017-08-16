/**
 * @file SpApp.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include <simpla/data/db/DataUtility.h>
#include "simpla/SIMPLA_config.h"

#include "SpApp.h"
#include "simpla/engine/Atlas.h"
#include "simpla/engine/Context.h"
#include "simpla/engine/Domain.h"
#include "simpla/engine/TimeIntegrator.h"
#include "simpla/geometry/Chart.h"
#include "simpla/parallel/MPIComm.h"
#include "simpla/parallel/Parallel.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/Logo.h"
#include "simpla/utilities/parse_command_line.h"

namespace simpla {
namespace application {
struct SpApp::pimpl_s {
    std::shared_ptr<engine::Schedule> m_schedule_ = nullptr;
    std::shared_ptr<engine::Context> m_context_ = nullptr;
    std::shared_ptr<engine::Atlas> m_atlas_ = nullptr;
};
SpApp::SpApp() : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_context_ = engine::Context::New();
    m_pimpl_->m_atlas_ = engine::Atlas::New();
}
SpApp::~SpApp() { delete m_pimpl_; };
void SpApp::Serialize(const std::shared_ptr<data::DataEntity> &cfg) const {
    base_type::Serialize(cfg);

    auto tdb = std::dynamic_pointer_cast<data::DataTable>(cfg);
    if (tdb != nullptr) {
        if (m_pimpl_->m_schedule_ != nullptr) { m_pimpl_->m_schedule_->Serialize(tdb->Get("Schedule")); }

        m_pimpl_->m_context_->Serialize(tdb->Get("Context"));
        m_pimpl_->m_atlas_->Serialize(tdb->Get("Atlas"));
    }
};
void SpApp::Deserialize(const std::shared_ptr<const data::DataEntity> &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataTable>(cfg);
    if (tdb != nullptr) {
        m_pimpl_->m_context_->Deserialize(tdb->Get("Context"));
        m_pimpl_->m_atlas_->Deserialize(tdb->Get("Atlas"));
        m_pimpl_->m_schedule_ = engine::Schedule::New(&cfg);
        m_pimpl_->m_schedule_->SetContext(m_pimpl_->m_context_);
        m_pimpl_->m_schedule_->SetAtlas(m_pimpl_->m_atlas_);
    }
    Click();
};

void SpApp::Config(int argc, char **argv) {
    std::string output_file = "h5://SimPlaOutput";

    std::string conf_file(argv[0]);
    std::string conf_prologue;
    std::string conf_epilogue;
    std::string app_name;
    conf_file += ".lua";

    auto cmd_line_cfg = data::DataTable::New();
    auto input_file_cfg = data::DataTable::New();

    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (opt == "i" || opt == "input") {
                input_file_cfg = data::DataTable::New(value);
            } else if (opt == "o" || opt == "output") {
                cmd_line_cfg->SetValue("OutputPath", value);
            } else if (opt == "log") {
                logger::open_file(value);
            } else if (opt == "v" || opt == "verbose") {
                logger::set_stdout_level(std::atoi(value.c_str()));
            } else if (opt == "quiet") {
                logger::set_stdout_level(logger::LOG_MESSAGE - 1);
            } else if (opt == "log_width") {
                logger::set_line_width(std::atoi(value.c_str()));
            } else if (opt == "version") {
                MESSAGE << "SIMPla " << ShowVersion();
                TheEnd(0);
                return TERMINATE;
            } else if (opt == "h" || opt == "help") {
                MESSAGE << std::endl
                        << ShowLogo() << std::endl
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
                        << ": Output file GetPrefix (default: simpla.h5)." << std::endl
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

                TheEnd(0);
                return TERMINATE;
            } else {
                cmd_line_cfg->SetValue(opt, value);
            }
            return CONTINUE;
        });
    MESSAGE << std::endl << ShowLogo() << std::endl;

    auto cfg = data::DataTable::New();

    cfg->Set("Context", input_file_cfg->Get("Context"));
    cfg->Set("Atlas", input_file_cfg->Get("Atlas"));
    cfg->Set("Schedule", input_file_cfg->Get("Schedule"));
    cfg->Set(*cmd_line_cfg);

    Deserialize(cfg);
}

void SpApp::DoInitialize() {
    m_pimpl_->m_context_->Initialize();
    if (m_pimpl_->m_schedule_ != nullptr) { m_pimpl_->m_schedule_->Initialize(); }
}
void SpApp::DoFinalize() {
    m_pimpl_->m_context_->Finalize();
    if (m_pimpl_->m_schedule_ != nullptr) {
        m_pimpl_->m_schedule_->Finalize();
        m_pimpl_->m_schedule_.reset();
    }
};
void SpApp::DoUpdate() {
    m_pimpl_->m_context_->Update();
    if (m_pimpl_->m_schedule_ != nullptr) { m_pimpl_->m_schedule_->Update(); };
};
void SpApp::DoTearDown() {
    m_pimpl_->m_context_->TearDown();
    if (m_pimpl_->m_schedule_ != nullptr) { m_pimpl_->m_schedule_->TearDown(); }
};
void SpApp::Run() {
    Update();
    if (m_pimpl_->m_schedule_ != nullptr) { m_pimpl_->m_schedule_->Run(); }
};

std::shared_ptr<engine::Context> SpApp::GetContext() const { return m_pimpl_->m_context_; }

void SpApp::SetSchedule(const std::shared_ptr<engine::Schedule> &s) {
    m_pimpl_->m_schedule_ = s;
    Click();
}
std::shared_ptr<engine::Schedule> SpApp::GetSchedule() const { return m_pimpl_->m_schedule_; }

}  // namespace application{
}  // namespace simpla{

using namespace simpla;
int main(int argc, char **argv) {
#ifndef NDEBUG
    logger::set_stdout_level(1000);
#endif

    parallel::init(argc, argv);

    MESSAGE << std::endl
            << data::DataBase::ShowDescription() << std::endl
            << engine::SPObject::ShowDescription() << std::endl;

    GLOBAL_COMM.barrier();

    auto app = application::SpApp::New();
    app->Initialize();

    if (GLOBAL_COMM.rank() == 0) {
        app->Config(argc, argv);
        std::ostringstream os;
        auto t_db = data::DataTable::New();
        app->Serialize(t_db);
        data::Pack(t_db, os, "lua");
        std::string buffer = os.str();
        parallel::bcast_string(&buffer);
    } else {
        std::string buffer;
        parallel::bcast_string(&buffer);
        auto t_cfg = data::DataTable::New("lua://");
        t_cfg->Set(buffer, nullptr);
        app->Deserialize(t_cfg);
    }

    app->Update();

    VERBOSE << DOUBLELINE << std::endl;
    VERBOSE << "SpApp:";
    //    app->Serialize(std::cout, 0);
    std::cout << std::endl;

    VERBOSE << DOUBLELINE << std::endl;
    GLOBAL_COMM.barrier();

    TheStart();
    app->Run();
    TheEnd();

    GLOBAL_COMM.barrier();
    VERBOSE << DOUBLELINE << std::endl;

    app->Finalize();

    parallel::close();
    logger::close();
}
