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
class Scenario;
class Atlas;
class Schedule : public engine::SPObject {
    SP_OBJECT_HEAD(Schedule, SPObject)

   public:
    SP_OBJECT_PROPERTY(size_type, MaxStep);
    SP_OBJECT_PROPERTY(size_type, CheckPointInterval);
    SP_OBJECT_PROPERTY(size_type, DumpInterval);

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void SetScenario(std::shared_ptr<Scenario> const& c) { m_Scenario_ = c; }
    std::shared_ptr<Scenario> GetScenario() const { return m_Scenario_; }

    void SetAtlas(std::shared_ptr<Atlas> const& a) { m_atlas_ = a; }
    std::shared_ptr<Atlas> GetAtlas() const { return m_atlas_; }

    index_box_type GetIndexBox() const;

    virtual void CheckPoint() const;
    virtual void Dump() const;

    size_type GetNumberOfStep() const;

    virtual void Synchronize();
    virtual void NextStep();
    virtual bool Done() const;

    void Run();
    std::shared_ptr<Scenario> m_Scenario_;
    std::shared_ptr<Atlas> m_atlas_;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
