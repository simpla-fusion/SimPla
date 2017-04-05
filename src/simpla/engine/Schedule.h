//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_SCHEDULE_H
#define SIMPLA_SCHEDULE_H

#include <memory>
#include "Task.h"
#include "simpla/data/all.h"
namespace simpla {
namespace engine {
class Schedule : public concept::Configurable {
   public:
    Schedule(std::shared_ptr<data::DataTable> const &t = nullptr);
    virtual ~Schedule();
    //    virtual std::shared_ptr<Task> NextTask() const;
    void AddTask(id_type mesh_id, std::shared_ptr<Task> const &);
    void RemoveTask(id_type);
    std::list<std::shared_ptr<Task>> const *GetTasks(id_type mesh_id) const;
    void RegisterAttributes(std::set<Attribute *> *) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
