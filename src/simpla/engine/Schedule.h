//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_SCHEDULE_H
#define SIMPLA_SCHEDULE_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/data/Data.h"

#include "Attribute.h"

namespace simpla {
namespace data {
class DataIOPort;
}
namespace engine {
class Context;
class Atlas;
class Schedule : public SPObject, public data::EnableCreateFromDataTable<Schedule> {
    SP_OBJECT_HEAD(Schedule, SPObject);

   public:
    explicit Schedule();
    ~Schedule() override;

    SP_DEFAULT_CONSTRUCT(Schedule)
    DECLARE_REGISTER_NAME(Schedule)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void SetContext(std::shared_ptr<Context> const &c) { m_ctx_ = c; }
    const std::shared_ptr<Context> GetContext() const { return m_ctx_; }
    std::shared_ptr<Context> GetContext() { return m_ctx_; }

    virtual void CheckPoint() const;
    virtual void Dump() const;

    size_type GetNumberOfStep() const;
    void SetMaxStep(size_type s);
    size_type GetMaxStep() const;

    virtual void Synchronize();
    virtual void NextStep();
    virtual bool Done() const;

    void Run();

    void SetCheckPointInterval(size_type s = 0);
    size_type GetCheckPointInterval() const;

    void SetDumpInterval(size_type s = 0);
    size_type GetDumpInterval() const;

   private:
    std::shared_ptr<Context> m_ctx_ = nullptr;
    std::shared_ptr<DataIOPort> m_data_io_ = nullptr;

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
