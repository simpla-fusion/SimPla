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
#include "simpla/engine/Model.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {

using namespace data;

template <typename THost>
class Maxwell {
    SP_ENGINE_POLICY_HEAD(Maxwell);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);
    void Advance(Real time_now, Real dt);

    Field<host_type, Real, CELL, 3> B0v{m_host_, "name"_ = "B0v"};

    Field<host_type, Real, FACE> B{m_host_, "name"_ = "B"};
    Field<host_type, Real, EDGE> E{m_host_, "name"_ = "E"};
    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};
    Field<host_type, Real, CELL, 3> dumpE{m_host_, "name"_ = "dumpE"};
    Field<host_type, Real, CELL, 3> dumpB{m_host_, "name"_ = "dumpB"};
    Field<host_type, Real, CELL, 3> dumpJ{m_host_, "name"_ = "dumpJ"};

    std::string m_boundary_geo_obj_prefix_ = "PEC";
};

template <typename TM>
std::shared_ptr<data::DataNode> Maxwell<TM>::Serialize() const {
    return nullptr;
};
template <typename TM>
void Maxwell<TM>::Deserialize(std::shared_ptr<const data::DataNode> cfg) {}

template <typename TM>
void Maxwell<TM>::InitialCondition(Real time_now) {
    dumpE.Clear();
    dumpB.Clear();
    dumpJ.Clear();
    E.Clear();
    B.Clear();
    J.Clear();

    B0v.Clear();

    if (m_host_->GetModel() != nullptr) { m_host_->GetModel()->LoadProfile("B0", &B0v); }
}
template <typename TM>
void Maxwell<TM>::BoundaryCondition(Real time_now, Real time_dt) {
    m_host_->FillBoundary(B, 0);
    m_host_->FillBoundary(E, 0);
    m_host_->FillBoundary(J, 0);
    //    m_host_->FillBoundary(dumpE, 0);
    //    m_host_->FillBoundary(dumpB, 0);
    //    m_host_->FillBoundary(dumpJ, 0);
}
template <typename TM>
void Maxwell<TM>::Advance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
    m_host_->FillBoundary(E, 0);

    B = B - curl(E) * dt;
    m_host_->FillBoundary(B, 0);

    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
    m_host_->FillBoundary(E, 0);

    //    dumpE.DeepCopy(E);
    //    dumpB.DeepCopy(B);
    //    dumpJ.DeepCopy(J);

    dumpE[0] = E.Get();
    dumpB[0] = B.Get();
    dumpJ[0] = J.Get();
    J.Clear();
}

}  // namespace simpla  {
#endif  // SIMPLA_MAXWELL_H
