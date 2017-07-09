//
// Created by salmon on 17-5-31.
//

#ifndef SIMPLA_SAMRAITIMEINTEGRATOR_H
#define SIMPLA_SAMRAITIMEINTEGRATOR_H
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
namespace simpla {

/**
* class SAMRAITimeIntegrator
*/
struct SAMRAITimeIntegrator : public engine::TimeIntegrator {
    SP_OBJECT_HEAD(SAMRAITimeIntegrator, engine::TimeIntegrator);

   public:
    SAMRAITimeIntegrator();
    ~SAMRAITimeIntegrator() override;
    SP_DEFAULT_CONSTRUCT(SAMRAITimeIntegrator)
    DECLARE_REGISTER_NAME(SAMRAITimeIntegrator)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> cfg) override;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void Synchronize() override;
    Real Advance(Real time_dt) override;
    bool Done() const override;

    void CheckPoint() const override;
    void Dump() const override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace simpla{
#endif  // SIMPLA_SAMRAITIMEINTEGRATOR_H
