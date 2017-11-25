//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Engine.h"
#include "simpla/physics/Field.h"
#include "simpla/physics/PhysicalConstants.h"
#include "simpla/utilities/Log.h"

namespace simpla {
namespace domain {

using namespace engine;

/**
 *  @ingroup FieldSolver
 *  @brief absorb boundary condition, PML
 */
template <typename TDomain>
class PML : public TDomain {
    SP_DOMAIN_HEAD(PML, TDomain);
    box_type m_center_box_ = {{0, 0, 0}, {1, 1, 1}};

   public:
    void SetCenterBox(box_type const& b) { m_center_box_ = b; }
    box_type GetCenterBox() const { return m_center_box_; }
    bool CheckBlockInBoundary() const override;

    FIELD(E, Real, EDGE);
    FIELD(B, Real, FACE);

    FIELD(X10, Real, EDGE);
    FIELD(X11, Real, EDGE);
    FIELD(X12, Real, EDGE);

    FIELD(X20, Real, FACE);
    FIELD(X21, Real, FACE);
    FIELD(X22, Real, FACE);

    // alpha
    FIELD(a0, Real, NODE, "LOCAL"_);
    FIELD(a1, Real, NODE, "LOCAL"_);
    FIELD(a2, Real, NODE, "LOCAL"_);
    // sigma
    FIELD(s0, Real, NODE, "LOCAL"_);
    FIELD(s1, Real, NODE, "LOCAL"_);
    FIELD(s2, Real, NODE, "LOCAL"_);

    FIELD(dX1, Real, EDGE, "LOCAL"_);
    FIELD(dX2, Real, FACE, "LOCAL"_);

    SP_PROPERTY(Real, dB);
    SP_PROPERTY(Real, expN);

   private:
    //    Real dB = 100, expN = 2;
};
template <typename TDomain>
PML<TDomain>::PML() : base_type(), m_dB_(100.0), m_expN_(2.0) {}
template <typename TDomain>
PML<TDomain>::~PML() {}

template <typename TDomain>
bool PML<TDomain>::CheckBlockInBoundary() const {
    return !geometry::isInSide(m_center_box_, this->GetBlockBox());
}

template <typename TDomain>
void PML<TDomain>::DoSetUp() {}
template <typename TDomain>
void PML<TDomain>::DoUpdate() {}
template <typename TDomain>
void PML<TDomain>::DoTearDown() {}
template <typename TDomain>
void PML<TDomain>::DoTagRefinementCells(Real time_now){};

template <typename TM>
void PML<TM>::DoInitialCondition(Real time_now) {
    X10.Clear();
    X11.Clear();
    X12.Clear();
    X20.Clear();
    X21.Clear();
    X22.Clear();
    E.Clear();
    B.Clear();
    dX1.Clear();
    dX2.Clear();

    a0.Clear();
    a1.Clear();
    a2.Clear();
    s0.Clear();
    s1.Clear();
    s2.Clear();
    DEFINE_PHYSICAL_CONST
    point_type c_xmin, c_xmax;
    std::tie(c_xmin, c_xmax) = m_center_box_;

    auto chart = this->GetChart();

#define sigma_(r) ((0.5 * (m_expN_ + 2.0) * 0.1 * m_dB_ * std::pow((r), m_expN_ + 1.0)))
#define alpha_(r) (1.0 + 2.0 * std::pow((r), m_expN_))
#define DEF(_N_)                                                                                      \
    {                                                                                                 \
        auto idx_box = a##_N_.GetShape();                                                             \
        point_type m_xmin = chart->local_coordinates(std::get<0>(idx_box));                           \
        point_type m_xmax = chart->local_coordinates(std::get<1>(idx_box));                           \
        a##_N_ = [&](point_type const& x) {                                                           \
            Real res = 1;                                                                             \
            if (x[_N_] <= c_xmin[_N_]) {                                                              \
                res = alpha_((c_xmin[_N_] - x[_N_]) / (c_xmin[_N_] - m_xmin[_N_]));                   \
            } else if (x[_N_] >= c_xmax[_N_]) {                                                       \
                res = alpha_((x[_N_] - c_xmax[_N_]) / (m_xmax[_N_] - c_xmax[_N_]));                   \
            }                                                                                         \
            return res;                                                                               \
        };                                                                                            \
    }                                                                                                 \
    {                                                                                                 \
        auto idx_box = s##_N_.GetShape();                                                             \
        point_type m_xmin = chart->local_coordinates(std::get<0>(idx_box));                           \
        point_type m_xmax = chart->local_coordinates(std::get<1>(idx_box));                           \
        s##_N_ = [&](point_type const& x) {                                                           \
            Real res = 0;                                                                             \
            if (x[_N_] <= c_xmin[_N_]) {                                                              \
                res = sigma_((c_xmin[_N_] - x[_N_]) / (c_xmin[_N_] - m_xmin[_N_])) * speed_of_light / \
                      (c_xmin[_N_] - m_xmin[_N_]);                                                    \
            } else if (x[_N_] >= c_xmax[_N_]) {                                                       \
                res = sigma_((x[_N_] - c_xmax[_N_]) / (m_xmax[_N_] - c_xmax[_N_])) * speed_of_light / \
                      (m_xmax[_N_] - c_xmax[_N_]);                                                    \
            }                                                                                         \
            return res;                                                                               \
        };                                                                                            \
    }

    DEF(0)
    DEF(1)
    DEF(2)
#undef DEF
}

template <typename TM>
void PML<TM>::DoAdvance(Real time_now, Real time_dt) {
    DEFINE_PHYSICAL_CONST

    dX2 = (X20 * (-2.0 * time_dt * s0) + curl_pdx(E) * time_dt) / (a0 + s0 * time_dt);
    X20 += dX2;
    B -= dX2;

    dX2 = (X21 * (-2.0 * time_dt * s0) + curl_pdy(E) * time_dt) / (a1 + s1 * time_dt);
    X21 += dX2;
    B -= dX2;

    dX2 = (X22 * (-2.0 * time_dt * s0) + curl_pdz(E) * time_dt) / (a2 + s2 * time_dt);
    X22 += dX2;
    B -= dX2;

    dX1 = (X10 * (-2.0 * time_dt * s0) + curl_pdx(B) / (mu0 * epsilon0) * time_dt) / (a0 + s0 * time_dt);
    X10 += dX1;
    E += dX1;

    dX1 = (X11 * (-2.0 * time_dt * s0) + curl_pdy(B) / (mu0 * epsilon0) * time_dt) / (a1 + s1 * time_dt);
    X11 += dX1;
    E += dX1;

    dX1 = (X12 * (-2.0 * time_dt * s0) + curl_pdz(B) / (mu0 * epsilon0) * time_dt) / (a2 + s2 * time_dt);
    X12 += dX1;
    E += dX1;
}
}  //    namespace  domain{
}  // namespace simpla

#endif  // SIMPLA_PML_H
