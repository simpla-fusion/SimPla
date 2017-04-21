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
    SP_OBJECT_HEAD(Schedule, SPObject);

   public:
    Schedule();
    virtual ~Schedule();
    Schedule(Schedule const &other) = delete;
    Schedule(Schedule &&other) = delete;
    Schedule &operator=(Schedule const &other) = delete;
    Schedule &operator=(Schedule &&other) = delete;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    void Initialize() override;
    void Finalize() override;
    void SetUp() override;
    void TearDown() override;

    virtual void Synchronize();
    virtual void NextStep();
    virtual bool Done() const;

    void Run();

    void SetOutputURL(std::string const &url);
    std::string const &GetOutputURL() const;

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
