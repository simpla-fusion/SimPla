//
// Created by salmon on 17-5-30.
//

#ifndef SIMPLA_EXTRASOURCE_H
#define SIMPLA_EXTRASOURCE_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/all.h"
#include "simpla/engine/all.h"
#include "simpla/physics/PhysicalConstants.h"

namespace simpla {
using namespace algebra;
using namespace data;
using namespace engine;

template <typename TM>
class ExtraSource : public engine::Domain {
    SP_OBJECT_HEAD(ExtraSource<TM>, engine::Domain)

   public:
    DOMAIN_HEAD(ExtraSource, TM)()(<#initializer#>)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<DataTable> cfg) override;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;

    std::shared_ptr<Attribute> m_attrr_;
};

template <typename TM>
bool ExtraSource<TM>::is_registered = engine::Domain::RegisterCreator<ExtraSource<TM>>();

template <typename TM>
std::shared_ptr<data::DataTable> ExtraSource<TM>::Serialize() const {
    auto res = std::make_shared<data::DataTable>();

    return res;
};
template <typename TM>
void ExtraSource<TM>::Deserialize(std::shared_ptr<DataTable> cfg) {
    DoInitialize();
    if (cfg == nullptr || cfg->GetTable("Species") == nullptr) { return; }

    Click();
}

template <typename TM>
void ExtraSource<TM>::DoInitialCondition(Real time_now) {
    Domain::DoInitialCondition(time_now);
}
template <typename TM>
void ExtraSource<TM>::DoBoundaryCondition(Real time_now, Real dt) {}
template <typename TM>
void ExtraSource<TM>::DoAdvance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST
}

}  // namespace simpla;
#endif  // SIMPLA_EXTRASOURCE_H
