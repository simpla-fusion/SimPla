//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include "Schedule.h"
#include "simpla/data/Serializable.h"
namespace simpla {
namespace engine {
class Context;
class TimeIntegratorBackend;
class TimeIntegrator : public Schedule, public data::Serializable {
    SP_OBJECT_BASE(engine::TimeIntegrator);

   public:
    TimeIntegrator(std::string const &s_engine = "", std::shared_ptr<Context> const &ctx = nullptr);
    TimeIntegrator(std::shared_ptr<data::DataTable> const &t, std::shared_ptr<Context> const &ctx = nullptr);
    ~TimeIntegrator();

    std::shared_ptr<Context> &GetContext();
    std::shared_ptr<Context> const &GetContext() const;

    void Initialize();
    void Finalize();
    void NextTimeStep(Real dt);

    bool RemainingSteps() const;
    Real CurrentTime() const;

   private:
    std::shared_ptr<TimeIntegratorBackend> m_backend_;
};

}  //{ namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
