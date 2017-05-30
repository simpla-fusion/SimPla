//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/physics/PhysicalConstants.h>
#include <simpla/utilities/Log.h>

namespace simpla {
using namespace engine;

/**
 *  @ingroup FieldSolver
 *  @brief absorb boundary condition, PML
 */
template <typename TChart>
class PML : public engine::Domain {
    SP_OBJECT_HEAD(PML<TChart>, engine::Worker);

   public:
    typedef engine::MeshView<TChart> mesh_type;
    typedef algebra::traits::scalar_type_t<mesh_type> scalar_type;

    template <int IFORM, int DOF = 1>
    using field_type = Field<mesh_type, scalar_type, IFORM, DOF>;

    explicit PML() : Domain(<#initializer#>, <#initializer#>) {};
    virtual ~PML(){};

    mesh_type m_mesh_;
    std::shared_ptr<MeshBase> GetMesh() { return &m_mesh_; }
    MeshBase const* GetMesh() const { return &m_mesh_; }

    void SetCenterDomain(geometry::GeoObject const&) {}

    void AdvanceData(Real data_time, Real dt = 0);
    void SetUp(Real time_now = 0);

     //    virtual std::string getClassName() const { return class_name(); }
    //    static std::string class_name() { return "PML<" + traits::type_id<TM>::GetPrefix() + ">"; }

    field_type<EDGE> E{m_mesh_, "name"_ = "E"};
    field_type<FACE> B{m_mesh_, "name"_ = "B"};

    field_type<EDGE> X10{m_mesh_}, X11{m_mesh_}, X12{m_mesh_};
    field_type<FACE> X20{m_mesh_}, X21{m_mesh_}, X22{m_mesh_};

    // alpha
    field_type<VERTEX> a0{m_mesh_};
    field_type<VERTEX> a1{m_mesh_};
    field_type<VERTEX> a2{m_mesh_};
    // sigma
    field_type<VERTEX> s0{m_mesh_};
    field_type<VERTEX> s1{m_mesh_};
    field_type<VERTEX> s2{m_mesh_};

    field_type<EDGE> dX1{m_mesh_};
    field_type<FACE> dX2{m_mesh_};

   private:
    inline Real sigma_(Real r, Real expN, Real dB) {
        return static_cast<Real>((0.5 * (expN + 2.0) * 0.1 * dB * std::pow(r, expN + 1.0)));
    }

    inline Real alpha_(Real r, Real expN, Real dB) { return static_cast<Real>(1.0 + 2.0 * std::pow(r, expN)); }
};

template <typename TM>
void PML<TM>::SetUp(Real time_now) {
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
    Real dB = 100, expN = 2;
    point_type m_xmin, m_xmax;
    point_type c_xmin, c_xmax;

    //    std::tie(m_xmin, m_xmax) = GetMesh()->box();
    //    std::tie(c_xmin, c_xmax) = center_box;
    //    auto dims = GetMesh()->dimensions();
    //
    //    GetMesh()->GetRange(VERTEX, mesh::SP_ES_ALL).foreach ([&](id_type const &s) {
    //        point_type x = m->point(s);
    //
    //#define DEF(_N_)                                                                            \
//    a##_N_[s] = 1;                                                                          \
//    s##_N_[s] = 0;                                                                          \
//    if (dims[_N_] > 1) {                                                                    \
//        if (x[_N_] < c_xmin[_N_]) {                                                         \
//            Real r = (c_xmin[_N_] - x[_N_]) / (m_xmax[_N_] - m_xmin[_N_]);                  \
//            a##_N_[s] = alpha_(r, expN, dB);                                                \
//            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (m_xmax[_N_] - m_xmin[_N_]); \
//        } else if (x[_N_] > c_xmax[_N_]) {                                                  \
//            Real r = (x[_N_] - c_xmax[_N_]) / (m_xmax[_N_] - m_xmin[_N_]);                  \
//            a##_N_[s] = alpha_(r, expN, dB);                                                \
//            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (m_xmax[_N_] - m_xmin[_N_]); \
//        }                                                                                   \
//    }
    //        DEF(0)
    //        DEF(1)
    //        DEF(2)
    //#undef DEF
    //    });
}

template <typename TM>
void PML<TM>::AdvanceData(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    dX2 = (X20 * (-2.0 * dt * s0) + curl_pdx(E) * dt) / (a0 + s0 * dt);
    X20 += dX2;
    B -= dX2;

    dX2 = (X21 * (-2.0 * dt * s0) + curl_pdy(E) * dt) / (a1 + s1 * dt);
    X21 += dX2;
    B -= dX2;

    dX2 = (X22 * (-2.0 * dt * s0) + curl_pdz(E) * dt) / (a2 + s2 * dt);
    X22 += dX2;
    B -= dX2;

    dX1 = (X10 * (-2.0 * dt * s0) + curl_pdx(B) / (mu0 * epsilon0) * dt) / (a0 + s0 * dt);
    X10 += dX1;
    E += dX1;

    dX1 = (X11 * (-2.0 * dt * s0) + curl_pdy(B) / (mu0 * epsilon0) * dt) / (a1 + s1 * dt);
    X11 += dX1;
    E += dX1;

    dX1 = (X12 * (-2.0 * dt * s0) + curl_pdz(B) / (mu0 * epsilon0) * dt) / (a2 + s2 * dt);
    X12 += dX1;
    E += dX1;
}
}  // namespace simpla

#endif  // SIMPLA_PML_H
