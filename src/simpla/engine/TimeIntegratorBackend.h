//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_TIMEINTEGRATORBACKEND_H
#define SIMPLA_TIMEINTEGRATORBACKEND_H

#include "SPObject.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/concept/Serializable.h"
namespace simpla {
namespace engine {
class Context;
class TimeIntegratorBackend : public concept::Serializable<TimeIntegratorBackend> {
    SP_OBJECT_BASE(TimeIntegratorBackend);

   public:
    TimeIntegratorBackend(std::shared_ptr<Context> const &ctx = nullptr);
    virtual ~TimeIntegratorBackend();

    void SetContext(std::shared_ptr<Context> const &ctx);
    std::shared_ptr<Context> &GetContext();
    std::shared_ptr<Context> const &GetContext() const;

    virtual void Initialize() = 0;
    virtual void Finalize() = 0;
    virtual void NextTimeStep(Real dt) = 0;
    virtual bool RemainingSteps() const = 0;
    virtual Real CurrentTime() const = 0;
    virtual size_type StepNumber() const = 0;

   private:
    std::shared_ptr<Context> m_ctx_;
};

struct DummyTimeIntegratorBackend : public TimeIntegratorBackend {
    SP_OBJECT_HEAD(DummyTimeIntegratorBackend, TimeIntegratorBackend);

   public:
    DummyTimeIntegratorBackend(){};
    virtual ~DummyTimeIntegratorBackend() {}

    virtual void Initialize(){};
    virtual void Finalize(){};
    virtual void NextTimeStep(Real dt){};
    virtual bool RemainingSteps() const { return 0; };
    virtual Real CurrentTime() const { return 0.0; };
    virtual size_type StepNumber() const { return 0; };
};
//
//struct TimeIntegratorBackendFactory {
//   public:
//    TimeIntegratorBackendFactory();
//    ~TimeIntegratorBackendFactory();
//    bool RegisterCreator(std::string const &k, std::function<TimeIntegratorBackend *()> const &);
//    template <typename U>
//    bool RegisterCreator(std::string const &k) {
//        return RegisterCreator(k, [&]() { return new U; });
//    }
//
//    TimeIntegratorBackend *Create(std::shared_ptr<data::DataTable> const &p);
//
//   private:
//    std::map<std::string, std::function<TimeIntegratorBackend *()>> m_TimeIntegrator_factory_;
//};
//#define GLOBAL_TIME_INTEGRATOR_FACTORY SingletonHolder<simpla::engine::TimeIntegratorBackendFactory>::instance()
}  // namespace engine{
}  // namespace simpla{

#endif  // SIMPLA_TIMEINTEGRATORBACKEND_H
