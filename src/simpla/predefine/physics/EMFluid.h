/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Domain.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {

using namespace data;

template <typename TDomain>
class EMFluid : public TDomain {
    SP_DOMAIN_HEAD(EMFluid, TDomain);

    Field<this_type, Real, CELL> ne{this, "Name"_ = "ne"};
    Field<this_type, Real, CELL, 3> B0v{this, "Name"_ = "B0v", "CheckPoint"_};

    Field<this_type, Real, EDGE> E0{this, "Name"_ = "E0"};
    Field<this_type, Real, FACE> B0{this, "Name"_ = "B0"};
    Field<this_type, Real, CELL> BB{this, "Name"_ = "BB"};
    Field<this_type, Real, CELL, 3> Jv{this, "Name"_ = "Jv"};
    Field<this_type, Real, CELL, 3> Ev{this, "Name"_ = "Ev"};
    Field<this_type, Real, CELL, 3> Bv{this, "Name"_ = "Bv"};
    Field<this_type, Real, CELL, 3> dE{this, "Name"_ = "dE"};
    Field<this_type, Real, FACE> B{this, "Name"_ = "B"};
    Field<this_type, Real, EDGE> E{this, "Name"_ = "E"};
    Field<this_type, Real, EDGE> J{this, "Name"_ = "J"};
    Field<this_type, Real, CELL, 3> dumpE{this, "Name"_ = "dumpE"};
    Field<this_type, Real, CELL, 3> dumpB{this, "Name"_ = "dumpB"};
    Field<this_type, Real, CELL, 3> dumpJ{this, "Name"_ = "dumpJ"};

    //    void TagRefinementCells(Real time_now);

    struct fluid_s {
        Real mass = 1;
        Real charge = 1;
        Real ratio = 1;
        std::shared_ptr<Field<this_type, Real, CELL>> n;
        std::shared_ptr<Field<this_type, Real, CELL, 3>> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;
    std::shared_ptr<fluid_s> AddSpecies(std::string const& name, std::shared_ptr<data::DataNode> d);
    std::map<std::string, std::shared_ptr<fluid_s>>& GetSpecies() { return m_fluid_sp_; };
};

template <typename TM>
std::shared_ptr<data::DataNode> EMFluid<TM>::Serialize() const {
    auto res = data::DataNode::New();
    for (auto& item : m_fluid_sp_) {
        res->SetValue("Species/" + item.first + "/mass", item.second->mass / SI_proton_mass);
        res->SetValue("Species/" + item.first + "/Z", item.second->charge / SI_elementary_charge);
        res->SetValue("Species/" + item.first + "/ratio", item.second->ratio);
    }
    return res;
};
template <typename TM>
void EMFluid<TM>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    cfg->Get("Species")->Foreach(
        [&](std::string const& k, std::shared_ptr<data::DataNode> v) { return AddSpecies(k, v) != nullptr ? 1 : 0; });
}

template <typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s> EMFluid<TM>::AddSpecies(std::string const& name,
                                                                     std::shared_ptr<data::DataNode> d) {
    if (d == nullptr) { return nullptr; }

    auto sp = std::make_shared<fluid_s>();
    sp->mass = d->GetValue<double>("mass", d->GetValue<double>("mass", 1)) * SI_proton_mass;
    sp->charge = d->GetValue<double>("charge", d->GetValue<double>("Z", 1)) * SI_elementary_charge;
    sp->ratio = d->GetValue<double>("ratio", d->GetValue<double>("ratio", 1));

    sp->n = std::make_shared<Field<this_type, Real, CELL>>(this, "Name"_ = name + "_n");
    sp->J = std::make_shared<Field<this_type, Real, CELL, 3>>(this, "Name"_ = name + "_J");
    m_fluid_sp_.emplace(name, sp);
    VERBOSE << "AddEntity particle : {\"" << name << "\", mass = " << sp->mass / SI_proton_mass
            << " [m_p], charge = " << sp->charge / SI_elementary_charge << " [q_e] }" << std::endl;
    return sp;
}
//
// template <typename TM>
// void EMFluid<TM>::TagRefinementCells(Real time_now) {
//    m_domain_->GetMesh()->TagRefinementCells(m_domain_->GetMesh()->GetRange(m_domain_->GetName() + "_BOUNDARY_3"));
//}
template <typename TM>
void EMFluid<TM>::DoSetUp() {}
template <typename TM>
void EMFluid<TM>::DoTearDown() {}
template <typename TM>
void EMFluid<TM>::DoInitialCondition(Real time_now) {
    E.Clear();
    B.Clear();
    J.Clear();

    Ev.Clear();
    Bv.Clear();

    ne.Clear();
    BB = dot(B0v, B0v);

    for (auto& item : m_fluid_sp_) {
        if (item.second == nullptr) { continue; }
        item.second->n->Clear();
        *item.second->n = ne * item.second->ratio;
        item.second->J->Clear();
    }
    Ev = map_to<CELL>(E);
}

template <typename TM>
void EMFluid<TM>::DoAdvance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    if (m_fluid_sp_.size() <= 0) { return; }
    Ev = map_to<CELL>(E);

    Field<this_type, Real, CELL, 3> Q{this};
    Field<this_type, Real, CELL, 3> K{this};

    Field<this_type, Real, CELL> a{this};
    Field<this_type, Real, CELL> b{this};
    Field<this_type, Real, CELL> c{this};

    a.Clear();
    b.Clear();
    c.Clear();

    Q.Clear();
    K.Clear();

    dE.Clear();

    for (auto& p : m_fluid_sp_) {
        Real ms = p.second->mass;
        Real qs = p.second->charge;
        auto& ns = *p.second->n;
        auto& Js = *p.second->J;

        auto as = static_cast<Real>((dt * qs) / (2.0 * ms));

        Q -= (0.5 * dt / epsilon0) * Js;

        K = Js + cross(Js, B0v) * as + Ev * ns * (qs * 2.0 * as);

        Js = (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);

        Q -= (0.5 * dt / epsilon0) * Js;

        a += qs * ns * (as / (BB * as * as + 1));
        b += qs * ns * (as * as / (BB * as * as + 1));
        c += qs * ns * (as * as * as / (BB * as * as + 1));
    }

    a *= 0.5 * dt / epsilon0;
    b *= 0.5 * dt / epsilon0;
    c *= 0.5 * dt / epsilon0;
    a += 1;

    dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

    for (auto& p : m_fluid_sp_) {
        Real ms = p.second->mass;
        Real qs = p.second->charge;
        auto& ns = *p.second->n;
        auto& Js = *p.second->J;

        auto as = static_cast<Real>((dt * qs) / (2.0 * ms));

        K = dE * ns * qs * as;

        Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
    }

    E = E + map_to<EDGE>(dE);
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
