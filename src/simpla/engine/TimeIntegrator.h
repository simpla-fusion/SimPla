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

struct TimeIntegrator : public Schedule {
    SP_OBJECT_HEAD(TimeIntegrator, Schedule);
    static bool is_register;

   public:
    TimeIntegrator(std::string const &k = "");
    ~TimeIntegrator();

    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable>);

    virtual void NextStep();
    virtual bool Done() const { return m_time_now_ >= m_time_end_ || Schedule::Done(); }

    virtual Real Advance(Real time_dt = 0.0);
    virtual void Synchronize();

    virtual void SetTime(Real t) { m_time_now_ = t; }
    virtual void SetTimeEnd(Real t) { m_time_end_ = t; }
    virtual void SetTimeStep(Real t) { m_time_step_ = t; };

    virtual Real GetTime() const { return m_time_now_; }
    virtual Real GetTimeEnd() const { return m_time_end_; }
    virtual Real GetTimeStep() const { return m_time_step_; }

    void SetCFL(Real c) { m_cfl_ = c; }
    Real GetCFL() const { return m_cfl_; }

   private:
    Real m_cfl_ = 0.9;
    Real m_time_now_ = 0.0;
    Real m_time_end_ = 1.0;
    Real m_time_step_ = 0.1;
};

}  //{ namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
