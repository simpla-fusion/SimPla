//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_SCHEDULE_H
#define SIMPLA_SCHEDULE_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "Attribute.h"
#include "SPObject.h"
#include "simpla/data/Data.h"

namespace simpla {
namespace data {
class DataIOPort;
}
namespace engine {
class Context;
class Atlas;
class Schedule : public engine::SPObject {
    SP_OBJECT_HEAD(Schedule, SPObject)

   public:
    SP_OBJECT_PROPERTY(size_type, SetMaxStep);
    SP_OBJECT_PROPERTY(size_type, CheckPointInterval);
    SP_OBJECT_PROPERTY(size_type, DumpInterval);

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void SetContext(const std::shared_ptr<Context> &c) { m_ctx_ = c; }
    std::shared_ptr<Context> GetContext() const { return m_ctx_; }

    virtual void CheckPoint() const;
    virtual void Dump() const;

    size_type GetNumberOfStep() const;

    virtual void Synchronize();
    virtual void NextStep();
    virtual bool Done() const;

    void Run();
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
