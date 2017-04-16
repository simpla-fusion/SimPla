//
// Created by salmon on 17-4-5.
//
#include "Schedule.h"
#include <simpla/parallel/all.h>
#include <simpla/toolbox/Logo.h>
#include <map>
#include <string>
#include "Atlas.h"
#include "Attribute.h"
#include "Context.h"
#include "Task.h"
#include "simpla/data/all.h"
namespace simpla {
namespace engine {
struct Schedule::pimpl_s {
    std::shared_ptr<Context> m_ctx_ = nullptr;

    size_type m_step_ = 0;
    size_type m_max_step_ = 0;
    size_type m_check_point_interval_ = 1;
    size_type m_dump_interval_ = 0;
    std::string m_output_url_ = "";
};
Schedule::Schedule() : m_pimpl_(new pimpl_s){};
Schedule::~Schedule(){};

size_type Schedule::GetNumberOfStep() const { return m_pimpl_->m_step_; }
void Schedule::SetMaxStep(size_type s) { m_pimpl_->m_max_step_ = s; }
size_type Schedule::GetMaxStep() const { return m_pimpl_->m_max_step_; }

void Schedule::SetCheckPointInterval(size_type s) { m_pimpl_->m_check_point_interval_ = s; }
size_type Schedule::GetCheckPointInterval() const { return m_pimpl_->m_check_point_interval_; }

void Schedule::SetDumpInterval(size_type s) { m_pimpl_->m_dump_interval_ = s; }
size_type Schedule::GetDumpInterval() const { return m_pimpl_->m_dump_interval_; }

void Schedule::NextStep() { ++m_pimpl_->m_step_; }
bool Schedule::Done() const { return m_pimpl_->m_max_step_ == 0 ? false : m_pimpl_->m_step_ >= m_pimpl_->m_max_step_; }

void Schedule::SetOutputURL(std::string const &url) { m_pimpl_->m_output_url_ = url; };
std::string const& Schedule::GetOutputURL() const { return m_pimpl_->m_output_url_; }

void Schedule::CheckPoint() const { UNIMPLEMENTED; }
void Schedule::Dump() const { UNIMPLEMENTED; }

void Schedule::Run() {
    while (!Done()) {
        Synchronize();
        NextStep();
        if (m_pimpl_->m_check_point_interval_ > 0 && m_pimpl_->m_step_ % m_pimpl_->m_check_point_interval_ == 0) {
            CheckPoint();
        };
        if (m_pimpl_->m_dump_interval_ > 0 && m_pimpl_->m_step_ % m_pimpl_->m_dump_interval_ == 0) { Dump(); };

        VERBOSE << " [ STEP:" << std::setw(5) << m_pimpl_->m_step_ << " ] " << std::endl;
    }
}

std::shared_ptr<data::DataTable> Schedule::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->Link("Context", m_pimpl_->m_ctx_->Serialize());
    return res;
}
void Schedule::Deserialize(std::shared_ptr<data::DataTable> t) {
    Initialize();
    m_pimpl_->m_ctx_->Deserialize(t->GetTable("Context"));
}

void Schedule::SetContext(std::shared_ptr<Context> ctx) { m_pimpl_->m_ctx_ = ctx; }

std::shared_ptr<Context> Schedule::GetContext() const { return m_pimpl_->m_ctx_; }
void Schedule::Initialize() { m_pimpl_->m_ctx_ = std::make_shared<Context>(); }
void Schedule::Finalize() { m_pimpl_->m_ctx_.reset(); }
void Schedule::SetUp() {
    m_pimpl_->m_ctx_->SetUp();
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
}
void Schedule::Synchronize() {
    //    auto &atlas = GetContext()->GetAtlas();
    //    if (from_level >= atlas.GetNumOfLevel() || to_level >= atlas.GetNumOfLevel()) { return; }
    //    for (auto const &src : atlas.Level(from_level)) {
    //        for (auto const &dest : atlas.Level(from_level)) {
    //            if (!geometry::CheckOverlap(src->GetIndexBox(), dest->GetIndexBox())) { continue; }
    //            //            auto s_it = m_pimpl_->m_patches_.find(src->GetGUID());
    //            //            auto d_it = m_pimpl_->m_patches_.find(dest->GetGUID());
    //            //            if (s_it == m_pimpl_->m_patches_.end() || d_it == m_pimpl_->m_patches_.end() || s_it ==
    //            d_it)
    //            //            { continue; }
    //            //            LOGGER << "Synchronize From " << m_pimpl_->m_atlas_.GetBlock(src)->GetIndexBox() << " to
    //            "
    //            //                   << m_pimpl_->m_atlas_.GetBlock(dest)->GetIndexBox() << " " << std::endl;
    //            //            auto &src_data = s_it->cast_as<data::DataTable>();
    //            //            src_data.Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const
    //            &dest_p)
    //            //            {
    //            //                auto dest_data = d_it->cast_as<data::DataTable>().Get(key);
    //            //                if (dest_data == nullptr) { return; }
    //            //            });
    //        }
    //    }
}

}  // namespace engine{
}  // namespace simpla{