//
// Created by salmon on 17-4-5.
//
#include "Schedule.h"
#include <map>
#include <string>
#include "Attribute.h"
#include "Task.h"
#include "simpla/data/all.h"
namespace simpla {
namespace engine {
struct Schedule::pimpl_s {
    bool m_isEnd_ = false;
    size_type m_step_counter_ = 0;
    std::shared_ptr<Task> m_first_task_;
    std::shared_ptr<Task> m_next_task_ = nullptr;
    std::map<std::string, std::shared_ptr<Attribute>> m_attributes_;
    std::map<id_type, std::list<std::shared_ptr<Task>>> m_tasks_;
};
Schedule::Schedule(std::shared_ptr<data::DataTable> const &t) : m_pimpl_(new pimpl_s), concept::Configurable(t){};
Schedule::~Schedule(){};

void Schedule::AddTask(id_type mesh_id, std::shared_ptr<Task> const &t) { m_pimpl_->m_tasks_[mesh_id].push_back(t); }
void Schedule::RemoveTask(id_type mesh_id) { m_pimpl_->m_tasks_.erase(mesh_id); };
std::list<std::shared_ptr<Task>> const *Schedule::GetTasks(id_type mesh_id) const {
    try {
        return &m_pimpl_->m_tasks_.at(mesh_id);
    } catch (...) { return nullptr; }
};

void Schedule::RegisterAttributes(std::set<Attribute *> *) const {}
// size_type Schedule::NumberOfSteps() const { return m_pimpl_->m_step_counter_; }

}  // namespace engine{
}  // namespace simpla{