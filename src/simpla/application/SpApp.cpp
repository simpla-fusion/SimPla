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
#include <simpla/geometry/Chart.h>
#include <simpla/parallel/all.h>
#include <simpla/utilities/Logo.h>
#include <simpla/utilities/parse_command_line.h>

using namespace simpla;

namespace simpla {
namespace application {
struct SpApp::pimpl_s {
    std::shared_ptr<engine::Schedule> m_schedule_ = nullptr;
    std::shared_ptr<engine::Context> m_context_ = nullptr;
};
SpApp::SpApp(std::string const &s_name) : SPObject(s_name), m_pimpl_(new pimpl_s) {}
SpApp::~SpApp() {}
std::shared_ptr<data::DataTable> SpApp::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    if (m_pimpl_->m_schedule_ != nullptr) { res->Set("Schedule", m_pimpl_->m_schedule_->Serialize()); }
    if (m_pimpl_->m_context_ != nullptr) { res->Set("Context", m_pimpl_->m_context_->Serialize()); }

    return res;
};
void SpApp::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    m_pimpl_->m_schedule_ = engine::Schedule::Create(cfg->Get("Schedule"));
    m_pimpl_->m_context_ = engine::Context::Create(cfg->Get("Context"));
    if (m_pimpl_->m_context_ == nullptr) {
        m_pimpl_->m_context_ = std::make_shared<engine::Context>();
        m_pimpl_->m_context_->Deserialize(cfg->GetTable("Context"));
    }
};
void SpApp::Initialize() {}
void SpApp::Finalize(){};
void SpApp::Update() {
    ASSERT(m_pimpl_->m_schedule_ != nullptr);
    m_pimpl_->m_context_->DoUpdate();
    m_pimpl_->m_schedule_->SetContext(m_pimpl_->m_context_);
    m_pimpl_->m_schedule_->DoUpdate();
};
void SpApp::TearDown() {
    if (m_pimpl_->m_schedule_ != nullptr) {
        m_pimpl_->m_schedule_->DoFinalize();
        m_pimpl_->m_schedule_.reset();
    }
    if (m_pimpl_->m_context_ != nullptr) {
        m_pimpl_->m_context_->DoFinalize();
        m_pimpl_->m_context_.reset();
    }
};
void SpApp::Run() {
    DoUpdate();
    if (m_pimpl_->m_schedule_ != nullptr) { m_pimpl_->m_schedule_->Run(); }
};

void SpApp::SetContext(std::shared_ptr<engine::Context> s) { m_pimpl_->m_context_ = s; }
std::shared_ptr<engine::Context> SpApp::GetContext() const { return m_pimpl_->m_context_; }

void SpApp::SetSchedule(std::shared_ptr<engine::Schedule> s) { m_pimpl_->m_schedule_ = s; }
std::shared_ptr<engine::Schedule> SpApp::GetSchedule() const { return m_pimpl_->m_schedule_; }

}  // namespace application{
}  // namespace simpla{
// static const bool _every_thing_are_registered = engine::Context::is_registered &&      //
//                                                engine::MeshBase::is_registered &&     //
//                                                engine::Domain::is_registered &&       //
//                                                geometry::GeoObject::is_registered &&  //
//                                                geometry::Chart::is_registered;

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

    auto cmd_line_cfg = std::make_shared<data::DataTable>();
    auto input_file_cfg = std::make_shared<data::DataTable>();

    simpla::parse_cmd_line(  //
        argc, argv, [&](std::string const &opt, std::string const &value) -> int {
            if (opt == "i" || opt == "input") {
                input_file_cfg.reset(new data::DataTable(value));
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
    MESSAGE << ShowLogo() << std::endl;

    auto cfg = std::make_shared<data::DataTable>();
    cfg->Set("Schedule", input_file_cfg->Get("Schedule"));
    cfg->Set("Context", input_file_cfg->Get("Context"));
    cfg->Set(cmd_line_cfg, true);

    MPI_Barrier(GLOBAL_COMM.comm());

    auto app = std::make_shared<application::SpApp>();
    app->Initialize();

    if (GLOBAL_COMM.rank() == 0) {
        app->Deserialize(cfg);

        std::ostringstream os;
        data::Serialize(app->Serialize(), os, "lua");
        std::string buffer = os.str();
        parallel::bcast_string(&buffer);
    } else {
        std::string buffer;
        parallel::bcast_string(&buffer);
        auto t_cfg = std::make_shared<data::DataTable>("lua://");
        t_cfg->backend()->Parser(buffer);

        app->Deserialize(t_cfg);
    }

    VERBOSE << DOUBLELINE << std::endl;
    VERBOSE << "SpApp:";
    app->Serialize(std::cout, 0);
    std::cout << std::endl;

    app->Update();

    VERBOSE << DOUBLELINE << std::endl;
    MPI_Barrier(GLOBAL_COMM.comm());

    TheStart();
    app->Run();
    TheEnd();

    MPI_Barrier(GLOBAL_COMM.comm());
    VERBOSE << DOUBLELINE << std::endl;

    app->TearDown();
    app->Finalize();

    parallel::close();
    logger::close();
}
