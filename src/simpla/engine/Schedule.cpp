//
// Created by salmon on 17-4-5.
//
#include "Schedule.h"
#include <simpla/toolbox/Logo.h>
#include <map>
#include <string>
#include "Attribute.h"
#include "Context.h"
#include "Task.h"
#include "simpla/data/all.h"
namespace simpla {
namespace engine {
struct Schedule::pimpl_s {
    std::shared_ptr<Context> m_ctx_;
    size_type m_step_ = 0;
    size_type m_max_step_ = 1;
};
Schedule::Schedule() : m_pimpl_(new pimpl_s) { m_pimpl_->m_ctx_ = std::make_shared<Context>(); };
Schedule::~Schedule() { Finalize(); };

void Schedule::SetNumberOfSteps(size_type s) { m_pimpl_->m_max_step_ = s; }
void Schedule::NextStep() {}
bool Schedule::Done() const { return m_pimpl_->m_step_ >= m_pimpl_->m_max_step_; }
void Schedule::CheckPoint() { DO_NOTHING; }

void Schedule::Run() {
    while (!Done()) {
        Synchronize();
        NextStep();
        VERBOSE << " [ STEP:" << std::setw(5) << m_pimpl_->m_step_ << " ] " << std::endl;
        ++m_pimpl_->m_step_;
    }

    //        if (step % step_of_check_points == 0) {
    //        data::DataTable(output_file).Set(ctx.db()->GetTable("Patches")); };
}

std::shared_ptr<data::DataTable> Schedule::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    if (m_pimpl_->m_ctx_ != nullptr) { res->Link("Context", m_pimpl_->m_ctx_->Serialize()); }
    return res;
}
void Schedule::Deserialize(std::shared_ptr<data::DataTable> t) {
    if (m_pimpl_->m_ctx_ == nullptr) { m_pimpl_->m_ctx_ = std::make_shared<Context>(); }
    m_pimpl_->m_ctx_->Deserialize(t->GetTable("Context"));
}

void Schedule::SetContext(std::shared_ptr<Context> ctx) { m_pimpl_->m_ctx_ = ctx; }

std::shared_ptr<Context> Schedule::GetContext() const { return m_pimpl_->m_ctx_; }

void Schedule::Initialize() {}
void Schedule::Update() {}
void Schedule::Finalize() {}
void Schedule::Synchronize(int from_level, int to_level) {
    auto &atlas = GetContext()->GetAtlas();
    if (from_level >= atlas.GetNumOfLevels() || to_level >= atlas.GetNumOfLevels()) { return; }
    for (auto const &src : atlas.Level(from_level)) {
        for (auto const &dest : atlas.Level(from_level)) {
            if (!geometry::CheckOverlap(src->GetIndexBox(), dest->GetIndexBox())) { continue; }
            //            auto s_it = m_pimpl_->m_patches_.find(src->GetGUID());
            //            auto d_it = m_pimpl_->m_patches_.find(dest->GetGUID());
            //            if (s_it == m_pimpl_->m_patches_.end() || d_it == m_pimpl_->m_patches_.end() || s_it == d_it)
            //            { continue; }
            //            LOGGER << "Synchronize From " << m_pimpl_->m_atlas_.GetBlock(src)->GetIndexBox() << " to   "
            //                   << m_pimpl_->m_atlas_.GetBlock(dest)->GetIndexBox() << " " << std::endl;
            //            auto &src_data = s_it->cast_as<data::DataTable>();
            //            src_data.Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &dest_p)
            //            {
            //                auto dest_data = d_it->cast_as<data::DataTable>().Get(key);
            //                if (dest_data == nullptr) { return; }
            //            });
        }
    }
}

}  // namespace engine{
}  // namespace simpla{