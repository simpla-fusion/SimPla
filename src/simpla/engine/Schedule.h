//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_SCHEDULE_H
#define SIMPLA_SCHEDULE_H

#include <memory>
#include "Attribute.h"

#include "simpla/data/all.h"
namespace simpla {
namespace engine {
class Context;
class Worker;

class Schedule : public data::Serializable, public data::EnableCreateFromDataTable<Schedule> {
    SP_OBJECT_BASE(Schedule);

   public:
    Schedule();
    virtual ~Schedule();
    virtual bool isInitialized() const;
    virtual void Initialize();  //!< make object configurable;
    virtual void Finalize();    //!< Release all resource, make object unconfigurable
    virtual void Update();      //!< effectuate the configuration

    virtual void Synchronize(int from_level = 0, int to_level = 0);
    virtual void NextStep();
    virtual bool Done() const;
    virtual void CheckPoint() const;
    virtual void Dump() const;

    size_type GetNumberOfStep() const;
    void SetMaxStep(size_type s);
    size_type GetMaxStep() const;

    void SetCheckPointInterval(size_type s = 0);
    size_type GetCheckPointInterval() const;

    void SetDumpInterval(size_type s = 0);
    size_type GetDumpInterval() const;

    void Run();

    std::shared_ptr<data::DataTable> Serialize() const;
    void Deserialize(std::shared_ptr<data::DataTable>);

    void SetContext(std::shared_ptr<Context>);
    std::shared_ptr<Context> GetContext() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
