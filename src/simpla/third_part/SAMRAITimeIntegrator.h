//
// Created by salmon on 17-5-31.
//

#ifndef SIMPLA_SAMRAITIMEINTEGRATOR_H
#define SIMPLA_SAMRAITIMEINTEGRATOR_H
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/Algebra.h"
#include "simpla/data/Data.h"
#include "simpla/engine/TimeIntegrator.h"
namespace simpla {

/**
* class SAMRAITimeIntegrator
*/
struct SAMRAITimeIntegrator : public engine::TimeIntegrator {
    SP_OBJECT_HEAD(SAMRAITimeIntegrator, engine::TimeIntegrator);

   public:
    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void Synchronize() override;
    void Advance(Real time_now, Real time_dt) override;
    bool Done() const override;

    void CheckPoint() const override;
    void Dump() const override;
};

}  // namespace simpla{
#endif  // SIMPLA_SAMRAITIMEINTEGRATOR_H
