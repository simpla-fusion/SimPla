//
// Created by salmon on 17-8-10.
//

#ifndef SIMPLA_PICBORIS_H
#define SIMPLA_PICBORIS_H

#include <simpla/physics/particle/Particle.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Domain.h"
#include "simpla/engine/Model.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {

using namespace data;

template <typename THost>
class PICBoris {
    SP_ENGINE_POLICY_HEAD(PICBoris);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);
    void Advance(Real time_now, Real dt);

    int DOF = 7;

    Particle<host_type> ele{m_host_, "name"_ = "ele", "DOF"_ = 6};

    Field<host_type, Real, CELL> ne{m_host_, "name"_ = "ne"};
    Field<host_type, Real, CELL, 3> B0v{m_host_, "name"_ = "B0v"};

    Field<host_type, Real, EDGE> E0{m_host_, "name"_ = "E0"};
    Field<host_type, Real, FACE> B0{m_host_, "name"_ = "B0"};
    Field<host_type, Real, CELL> BB{m_host_, "name"_ = "BB"};
    Field<host_type, Real, CELL, 3> Jv{m_host_, "name"_ = "Jv"};

    Field<host_type, Real, FACE> B{m_host_, "name"_ = "B"};
    Field<host_type, Real, EDGE> E{m_host_, "name"_ = "E"};
    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};

    //    void TagRefinementCells(Real time_now);

    std::map<std::string, std::shared_ptr<Particle<THost>>> m_particle_sp_;
    std::shared_ptr<Particle<THost>> AddSpecies(std::string const& name, std::shared_ptr<data::DataNode> d);
    //    template <typename... Args>
    //    std::shared_ptr<Particle<THost>> AddSpecies(std::string const& name, Args&&... args) {
    //        data::DataNode t;
    ////        t.Assign(std::forward<Args>(args)...);
    //        return AddSpecies(name, t);
    //    };

    std::map<std::string, std::shared_ptr<Particle<THost>>>& GetSpecies() { return m_particle_sp_; };
};

template <typename TM>
std::shared_ptr<data::DataNode> PICBoris<TM>::Serialize() const {
    auto res = data::DataNode::New();
    for (auto& item : m_particle_sp_) {
        //        t.SetEntity<double>("mass", item.m_node_->mass / SI_proton_mass);
        //        t.SetEntity<double>("Z", item.m_node_->charge / SI_elementary_charge);
        //        t.SetEntity<double>("ratio", item.m_node_->ratio);

        res->Set(item.first, item.second->Serialize());
    }
    return res;
};
template <typename TM>
void PICBoris<TM>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {
    cfg->Get("Species")->Foreach(
        [&](std::string k, std::shared_ptr<data::DataNode> t) { return AddSpecies(k, t) != nullptr; });
}

template <typename TM>
std::shared_ptr<Particle<TM>> PICBoris<TM>::AddSpecies(std::string const& name,
                                                       std::shared_ptr<data::DataNode> d) {
    auto sp = Particle<TM>::New(m_host_, d);
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
//    m_mesh_->GetMesh()->TagRefinementCells(m_mesh_->GetMesh()->GetRange(m_mesh_->GetName() + "_BOUNDARY_3"));
//}
template <typename TM>
void PICBoris<TM>::InitialCondition(Real time_now) {}
template <typename TM>
void PICBoris<TM>::BoundaryCondition(Real time_now, Real time_dt) {}
template <typename TM>
void PICBoris<TM>::Advance(Real time_now, Real dt) {}

}  // namespace simpla  {

#endif  // SIMPLA_PICBORIS_H
