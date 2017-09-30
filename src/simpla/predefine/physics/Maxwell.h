/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_MAXWELL_H
#define SIMPLA_MAXWELL_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Domain.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {
namespace domain {

using namespace data;

template <typename TDomainBase>
class Maxwell : public TDomainBase {
    SP_DOMAIN_HEAD(Maxwell, TDomainBase);

    Field<this_type, Real, FACE> B{this, "Name"_ = "B", "CheckPoint"_};
    Field<this_type, Real, EDGE> E{this, "Name"_ = "E", "CheckPoint"_};
    Field<this_type, Real, EDGE> J{this, "Name"_ = "J", "CheckPoint"_};
};
template <typename TDomain>
Maxwell<TDomain>::Maxwell() : base_type() {}
template <typename TDomain>
Maxwell<TDomain>::~Maxwell() {}
template <typename TDomain>
std::shared_ptr<data::DataNode> Maxwell<TDomain>::Serialize() const {
    return base_type::Serialize();
};
template <typename TDomain>
void Maxwell<TDomain>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    base_type::Deserialize(cfg);
}
template <typename TDomain>
void Maxwell<TDomain>::DoSetUp() {}
template <typename TDomain>
void Maxwell<TDomain>::DoUpdate() {}

template <typename TDomain>
void Maxwell<TDomain>::DoTearDown() {}

template <typename TDomain>
void Maxwell<TDomain>::DoInitialCondition(Real time_now) {
    E.Clear();
    B.Clear();
    J.Clear();
}
template <typename TDomain>
void Maxwell<TDomain>::DoBoundaryCondition(Real time_now, Real time_dt) {
    //    this->FillBoundary(B, 0);
    //    this->FillBoundary(E, 0);
    //    this->FillBoundary(J, 0);
    //    m_domain_->FillBoundary(dumpE, 0);
    //    m_domain_->FillBoundary(dumpB, 0);
    //    m_domain_->FillBoundary(dumpJ, 0);
}

template <typename TDomain>
void Maxwell<TDomain>::DoAdvance(Real time_now, Real time_dt) {
    DEFINE_PHYSICAL_CONST

    //    B = curl(E);
    //    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * time_dt;
    //    //    this->FillBoundary(E, 0);
    //
    //    B = B - curl(E) * time_dt;
    //    //    this->FillBoundary(B, 0);
    //
    //    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * time_dt;
    //    //    this->FillBoundary(E, 0);

    J.Clear();
}

template <typename TDomain>
void Maxwell<TDomain>::DoTagRefinementCells(Real time_now){};

}  // namespace domain{
}  // namespace simpla  {
#endif  // SIMPLA_MAXWELL_H
