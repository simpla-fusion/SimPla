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
    SP_OBJECT_DECLARE_MEMBERS(Schedule, SPObject)
    static constexpr char const *TagName() { return "Schedule"; }

   public:
    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    template <typename TContext>
    std::shared_ptr<Context> NewContext() {
        m_ctx_ = Context::New();
        return m_ctx_;
    }
    void SetContext(const std::shared_ptr<Context> &c) { m_ctx_ = c; }
    std::shared_ptr<Context> GetContext() const { return m_ctx_; }

    void SetAtlas(const std::shared_ptr<Atlas> &a) { m_atlas_ = a; }
    std::shared_ptr<Atlas> GetAtlas() const { return m_atlas_; }

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
    std::shared_ptr<Atlas> m_atlas_ = nullptr;
    std::shared_ptr<data::DataIOPort> m_data_io_ = nullptr;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
