//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Configurable.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <memory>
#include "SPObject.h"
namespace simpla {
namespace engine {
class Context;
class TimeIntegrator : public concept::Configurable {
    SP_OBJECT_BASE(engine::TimeIntegrator);

   public:
    TimeIntegrator(std::shared_ptr<Context> const &ctx = nullptr, std::shared_ptr<data::DataTable> const &t = nullptr);
    virtual ~TimeIntegrator();
    void SetContext(std::shared_ptr<Context> const &);
    std::shared_ptr<Context> const &GetContext() const;

    virtual void Initialize();
    virtual void Finalize();

    virtual Real Advance(Real dt, int level = 0);
    virtual size_type NextTimeStep(Real dt);
    virtual size_type step() const;
    virtual bool remainingSteps() const;
    virtual Real timeNow() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

struct TimeIntegratorFactory {
   public:
    TimeIntegratorFactory();
    ~TimeIntegratorFactory();
    bool RegisterCreator(std::string const &k,
                         std::function<std::shared_ptr<TimeIntegrator>(
                             std::shared_ptr<Context> const &, std::shared_ptr<data::DataEntity> const &)> const &);
    template <typename U>
    bool RegisterCreator(std::string const &k) {
        return RegisterCreator(k, [&](std::shared_ptr<Context> const &m, std::shared_ptr<data::DataEntity> const &t) {
            return std::make_shared<U>(m, t);
        });
    }

    std::shared_ptr<TimeIntegrator> Create(std::string const &, std::shared_ptr<Context> const &m = nullptr,
                                           std::shared_ptr<data::DataTable> const &p = nullptr);

    std::shared_ptr<TimeIntegrator> Create(std::shared_ptr<data::DataTable> const &p = nullptr);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
#define GLOBAL_TIME_INTEGRATOR_FACTORY SingletonHolder<simpla::engine::TimeIntegratorFactory>::instance()
static int REGISTERED_TIME_INTEGRATOR = 0;
}  //{ namespace engine
}  // namespace simpla

#endif  // SIMPLA_TIMEINTEGRATOR_H
