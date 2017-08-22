//
// Created by salmon on 17-8-10.
//

#ifndef SIMPLA_PICBORIS_H
#define SIMPLA_PICBORIS_H

#include <simpla/particle/Particle.h>
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

    void Serialize(DataTable& cfg) const;
    void Deserialize(const DataTable& cfg);
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
    std::shared_ptr<Particle<THost>> AddSpecies(std::string const& name, data::DataTable const& d);
    //    template <typename... Args>
    //    std::shared_ptr<Particle<THost>> AddSpecies(std::string const& name, Args&&... args) {
    //        data::DataTable t;
    ////        t.Assign(std::forward<Args>(args)...);
    //        return AddSpecies(name, t);
    //    };

    std::map<std::string, std::shared_ptr<Particle<THost>>>& GetSpecies() { return m_particle_sp_; };
};

template <typename TM>
void PICBoris<TM>::Serialize(DataTable& cfg) const {
    for (auto& item : m_particle_sp_) {
        //        t.Set<double>("mass", item.second->mass / SI_proton_mass);
        //        t.Set<double>("Z", item.second->charge / SI_elementary_charge);
        //        t.Set<double>("ratio", item.second->ratio);

        item.second->Serialize(cfg.GetTable("Species/" + item.first));
    }
};
template <typename TM>
void PICBoris<TM>::Deserialize(const DataTable& cfg) {
    cfg.GetTable("Species").Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> v) {
        auto t = std::dynamic_pointer_cast<data::DataTable>(v);
        if (t != nullptr) {
            t->SetValue("name", k);
            AddSpecies(k, *t);
        }
        return 1;
    });
}

template <typename TM>
std::shared_ptr<Particle<TM>> PICBoris<TM>::AddSpecies(std::string const& name, data::DataTable const& d) {
    auto sp = Particle<TM>::New(m_host_, d);
    sp->SetDOF(7);
    sp->db().SetValue("mass", d.GetValue<double>("mass", d.GetValue<double>("mass", 1)) * SI_proton_mass);
    sp->db().SetValue("charge", d.GetValue<double>("charge", d.GetValue<double>("Z", 1)) * SI_elementary_charge);
    //    sp->ratio = d.Get<double>("ratio", d.Get<double>("ratio", 1));

    m_particle_sp_.emplace(name, sp);
    VERBOSE << "Add particle : {\" Name=" << name
            << "\", mass = " << sp->db().template GetValue<double>("mass") / SI_proton_mass
            << " [m_p], charge = " << sp->db().template GetValue<double>("charge") / SI_elementary_charge << " [q_e] }"
            << std::endl;
    return sp;
}
//
// template <typename TM>
// void PICBoris<TM>::TagRefinementCells(Real time_now) {
//    m_host_->GetMesh()->TagRefinementCells(m_host_->GetMesh()->GetRange(m_host_->GetName() + "_BOUNDARY_3"));
//}
template <typename TM>
void PICBoris<TM>::InitialCondition(Real time_now) {}
template <typename TM>
void PICBoris<TM>::BoundaryCondition(Real time_now, Real time_dt) {}
template <typename TM>
void PICBoris<TM>::Advance(Real time_now, Real dt) {}

}  // namespace simpla  {

#endif  // SIMPLA_PICBORIS_H
