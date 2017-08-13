//
// Created by salmon on 17-5-31.
//

#ifndef SIMPLA_SAMRAITIMEINTEGRATOR_H
#define SIMPLA_SAMRAITIMEINTEGRATOR_H
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/Algebra.h"
#include "simpla/data/Data.h"
#include "simpla/engine/Engine.h"
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

    void Serialize(data::DataTable &cfg) const override;
    void Deserialize(const data::DataTable &cfg) override;

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
