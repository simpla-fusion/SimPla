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
struct TimeIntegratorBackend : public data::Serializable,
                               public data::EnableCreateFromDataTable<TimeIntegratorBackend> {
    SP_OBJECT_BASE(engine::TimeIntegratorBackend);

    TimeIntegratorBackend();
    virtual ~TimeIntegratorBackend();
    virtual Real Advance(Context *ctx, Real time_now, Real time_dt);
    virtual void Synchronize(Context *ctx, int from_level = 0, int to_level = 0);
};
struct TimeIntegrator : public Schedule {
    SP_OBJECT_HEAD(TimeIntegrator,  Schedule);

   public:
    TimeIntegrator(std::string const &k = "");
    ~TimeIntegrator();
    std::shared_ptr<data::DataTable> Serialize() const;
    void Deserialize(std::shared_ptr<data::DataTable>);

    Real Advance(Real time_dt = 0.0);
    void Synchronize(int from_level = 0, int to_level = 0);

    void NextStep();
    bool Done() const { return m_time_now_ >= m_time_end_; }

    void SetTime(Real t) { m_time_now_ = t; }
    void SetTimeEnd(Real t) { m_time_end_ = t; }
    void SetTimeStep(Real t) { m_time_step_ = t; };

    Real GetTime() const { return m_time_now_; }
    Real GetTimeEnd() const { return m_time_end_; }
    Real GetTimeStep() const { return m_time_step_; }

   private:
    std::shared_ptr<TimeIntegratorBackend> m_backend_;

    Real m_time_now_ = 0.0;
    Real m_time_end_ = 1.0;
    Real m_time_step_ = 0.1;
};

}  //{ namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
