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
class Atlas;
class Schedule : public SPObject, public data::Serializable, public data::EnableCreateFromDataTable<Schedule> {
    SP_OBJECT_BASE(Schedule);

   public:
    Schedule();
    virtual ~Schedule();

    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable>);

    virtual void Initialize();
    virtual void Finalize();
    virtual void SetUp();
    virtual void TearDown();

    virtual void Synchronize();

    virtual void NextStep();

    virtual bool Done() const;

    void Run();

    void SetOutputURL(std::string const& url);
    std::string const& GetOutputURL() const;

    virtual void CheckPoint() const;
    virtual void Dump() const;

    size_type GetNumberOfStep() const;
    void SetMaxStep(size_type s);
    size_type GetMaxStep() const;

    void SetCheckPointInterval(size_type s = 0);
    size_type GetCheckPointInterval() const;

    void SetDumpInterval(size_type s = 0);
    size_type GetDumpInterval() const;

    void SetContext(std::shared_ptr<Context>);
    std::shared_ptr<Context> GetContext() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H