//
// Created by salmon on 17-8-10.
//

#ifndef SIMPLA_PICBORIS_H
#define SIMPLA_PICBORIS_H

#include <simpla/physics/particle/Particle.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Domain.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {

using namespace data;

template <typename TDomain>
class PICBoris : public TDomain {
    SP_DOMAIN_HEAD(PICBoris, TDomain);

   public:
    int DOF = 7;

    Particle<this_type> ele{this, "name"_ = "ele", "DOF"_ = 6};

    Field<this_type, Real, CELL> ne{this, "name"_ = "ne"};
    Field<this_type, Real, CELL, 3> B0v{this, "name"_ = "B0v"};

    Field<this_type, Real, EDGE> E0{this, "name"_ = "E0"};
    Field<this_type, Real, FACE> B0{this, "name"_ = "B0"};
    Field<this_type, Real, CELL> BB{this, "name"_ = "BB"};
    Field<this_type, Real, CELL, 3> Jv{this, "name"_ = "Jv"};

    Field<this_type, Real, FACE> B{this, "name"_ = "B"};
    Field<this_type, Real, EDGE> E{this, "name"_ = "E"};
    Field<this_type, Real, EDGE> J{this, "name"_ = "J"};

    //    void TagRefinementCells(Real time_now);

    std::map<std::string, std::shared_ptr<Particle<base_type>>> m_particle_sp_;
    std::shared_ptr<Particle<base_type>> AddSpecies(std::string const& name, std::shared_ptr<data::DataNode> d);
    std::map<std::string, std::shared_ptr<Particle<base_type>>>& GetSpecies() { return m_particle_sp_; };

    //    template <typename... Args>
    //    std::shared_ptr<Particle<base_type>> AddSpecies(std::string const& name, Args&&... args) {
    //        data::DataNode t;
    //        t.Assign(std::forward<Args>(args)...);
    //        return AddSpecies(name, t);
    //    };
};

template <typename TM>
std::shared_ptr<data::DataNode> PICBoris<TM>::Serialize() const {
    auto res = data::DataNode::New();
    for (auto& item : m_particle_sp_) { res->Set(item.first, item.second->Serialize()); }
    return res;
};
template <typename TM>
void PICBoris<TM>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    cfg->Get("Species")->Foreach(
        [&](std::string k, std::shared_ptr<data::DataNode> t) { return AddSpecies(k, t) != nullptr; });

    //        t.SetEntity<double>("mass", item.m_node_->mass / SI_proton_mass);
    //        t.SetEntity<double>("Z", item.m_node_->charge / SI_elementary_charge);
    //        t.SetEntity<double>("ratio", item.m_node_->ratio);
}

template <typename TM>
std::shared_ptr<Particle<TM>> PICBoris<TM>::AddSpecies(std::string const& name, std::shared_ptr<data::DataNode> d) {
    auto sp = Particle<TM>::New(this, d);
    sp->SetDOF(7);
    sp->db()->SetValue("mass", d->GetValue<double>("mass", d->GetValue<double>("mass", 1)) * SI_proton_mass);
    sp->db()->SetValue("charge", d->GetValue<double>("charge", d->GetValue<double>("Z", 1)) * SI_elementary_charge);
    //    sp->ratio = d->Get<double>("ratio", d->GetEntity<double>("ratio", 1));

    m_particle_sp_.emplace(name, sp);
    VERBOSE << "AddEntity particle : {\" Name=" << name
            << "\", mass = " << sp->db()->template GetValue<double>("mass") / SI_proton_mass
            << " [m_p], charge = " << sp->db()->template GetValue<double>("charge") / SI_elementary_charge << " [q_e] }"
            << std::endl;
    return sp;
}
//
// template <typename TM>
// void PICBoris<TM>::TagRefinementCells(Real time_now) {
//    m_domain_->GetMesh()->TagRefinementCells(m_domain_->GetMesh()->GetRange(m_domain_->GetName() + "_BOUNDARY_3"));
//}

template <typename TM>
void PICBoris<TM>::DoSetUp() {}
template <typename TM>
void PICBoris<TM>::DoUpdate() {}
template <typename TM>
void PICBoris<TM>::DoTearDown() {}

template <typename TM>
void PICBoris<TM>::DoInitialCondition(Real time_now) {}
template <typename TM>
void PICBoris<TM>::DoBoundaryCondition(Real time_now, Real dt) {}
template <typename TM>
void PICBoris<TM>::DoAdvance(Real time_now, Real dt) {}
template <typename TM>
void PICBoris<TM>::DoTagRefinementCells(Real time_now) {}

}  // namespace simpla  {

#endif  // SIMPLA_PICBORIS_H
