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
class Schedule : public SPObject, public data::EnableCreateFromDataTable<Schedule> {
    SP_OBJECT_HEAD(Schedule, SPObject);

   public:
    explicit Schedule(std::string const &s_name = "");
    ~Schedule() override;

    SP_DEFAULT_CONSTRUCT(Schedule)
    DECLARE_REGISTER_NAME("Schedule")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    void Initialize() override;
    void Finalize() override;
    void SetUp() override;
    void TearDown() override;

    virtual void Synchronize();
    virtual void NextStep();
    virtual bool Done() const;

    void Run();

    std::shared_ptr<Context> SetContext(std::shared_ptr<Context> const &ctx);
    std::shared_ptr<Context> const &GetContext() const;
    std::shared_ptr<Context> &GetContext();

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

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_SCHEDULE_H
