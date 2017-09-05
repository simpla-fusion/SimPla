//
// Created by salmon on 17-9-5.
//

#ifndef SIMPLA_SIMPLETIMEINTEGRATOR_H
#define SIMPLA_SIMPLETIMEINTEGRATOR_H

#include "simpla/engine/TimeIntegrator.h"
namespace simpla {
class SimpleTimeIntegrator : public engine::TimeIntegrator {
    //    SP_OBJECT_HEAD(SimpleTimeIntegrator, engine::TimeIntegrator);

   public:
    static std::string GetFancyTypeName_s() {
        return engine::TimeIntegrator::GetFancyTypeName_s() + "." + __STRING(_CLASS_NAME_);
    }
    virtual std::string GetFancyTypeName() const override { return GetFancyTypeName_s(); }
    static bool _is_registered;

   private:
    typedef engine::TimeIntegrator base_type;
    typedef SimpleTimeIntegrator this_type;
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;

   public:
   protected:
    SimpleTimeIntegrator();

   public:
    ~SimpleTimeIntegrator() override;

    std::shared_ptr<simpla::data::DataNode> Serialize() const override;
    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;

   private:
    template <typename U, typename... Args>
    static std::shared_ptr<U> TryNew(std::true_type, Args &&... args) {
        return std::shared_ptr<U>(new U(std::forward<Args>(args)...));
    };

    template <typename U>
    static std::shared_ptr<U> TryNew(std::false_type, std::shared_ptr<simpla::data::DataNode> const &cfg) {
        return std::dynamic_pointer_cast<U>(simpla::SPObject::Create(cfg));
    };

   public:
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return TryNew<this_type>(std::integral_constant<bool, !std::is_abstract<this_type>::value>(),
                                 std::forward<Args>(args)...);
    };
    static std::shared_ptr<this_type> New() {
        return TryNew<this_type>(std::integral_constant<bool, !std::is_abstract<this_type>::value>());
    };

   public:
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void Synchronize() override;

    void Advance(Real time_now, Real time_dt) override;
};
}  // namespace simpla
#endif  // SIMPLA_SIMPLETIMEINTEGRATOR_H
