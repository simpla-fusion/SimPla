//
// Created by salmon on 16-5-21.
//

#ifndef SIMPLA_PML_H
#define SIMPLA_PML_H

#include "../../src/gtl/Log.h"
#include "../../src/sp_def.h"
#include "../../src/physics/PhysicalConstants.h"
#include "../../src/field/Field.h"
#include "../../src/manifold/Calculus.h"
#include "../../src/simulation/Context.h"

namespace simpla
{
using namespace mesh;

/**
 *  @ingroup FieldSolver
 *  @brief absorb boundary condition, PML
 */
template<typename TM>
class PML : public simulation::ProblemDomain
{
    typedef simulation::ProblemDomain base_type;
public:
    typedef TM mesh_type;

    template<typename ValueType, size_t IFORM> using field_t =  Field <ValueType, TM, index_const<IFORM>>;;

    PML(const mesh_type *mp, box_type const &center_box);

    virtual ~PML();

    virtual void next_step(Real dt);

    virtual void deploy();

    virtual void sync(mesh::TransitionMap const &, simulation::ProblemDomain const &other);

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return "PML<" + traits::type_id<TM>::name() + ">"; }


    mesh_type const *m = nullptr;


    field_t<scalar_type, mesh::EDGE> E{m};
    field_t<scalar_type, mesh::FACE> B{m};


    field_t<scalar_type, mesh::EDGE> X10{m}, X11{m}, X12{m};
    field_t<scalar_type, mesh::FACE> X20{m}, X21{m}, X22{m};

    // alpha
    field_t<scalar_type, mesh::VERTEX> a0{m};
    field_t<scalar_type, mesh::VERTEX> a1{m};
    field_t<scalar_type, mesh::VERTEX> a2{m};
    // sigma
    field_t<scalar_type, mesh::VERTEX> s0{m};
    field_t<scalar_type, mesh::VERTEX> s1{m};
    field_t<scalar_type, mesh::VERTEX> s2{m};

    field_t<scalar_type, mesh::EDGE> dX1{m};
    field_t<scalar_type, mesh::FACE> dX2{m};


private:


    inline Real sigma_(Real r, Real expN, Real dB)
    {
        return static_cast<Real>((0.5 * (expN + 2.0) * 0.1 * dB * std::pow(r, expN + 1.0)));
    }

    inline Real alpha_(Real r, Real expN, Real dB)
    {
        return static_cast<Real>(1.0 + 2.0 * std::pow(r, expN));
    }


};


template<typename TM>
PML<TM>::~PML() { }

template<typename TM>
PML<TM>::PML(const mesh_type *mp, box_type const &center_box) : base_type(mp), m(mp)
{
    assert(mp != nullptr);
//
//    properties()["DISABLE_WRITE"] = false;
//
//    int od[3] = {0, 0, 0};
//
//    if (p_od != nullptr)
//    {
//        od[0] = p_od[0];
//        od[1] = p_od[1];
//        od[2] = p_od[2];
//    }
//
//    point_type xmin, xmax;
//
//    std::tie(xmin, xmax) = m->box();
//
//    LOGGER << "create PML solver [" << xmin << " , " << xmax << " ]" << std::endl;
//
//    DEFINE_PHYSICAL_CONST
//
//    Real dB = 100, expN = 2;
//
//    a0.clear();
//    a1.clear();
//    a2.clear();
//    s0.clear();
//    s1.clear();
//    s2.clear();
//    X10.clear();
//    X11.clear();
//    X12.clear();
//    X20.clear();
//    X21.clear();
//    X22.clear();
//
//    point_type ymin, ymax;
//    std::tie(ymin, ymax) = m->box();
//
//    m->range(mesh::VERTEX, SP_ES_VALID).foreach(
//            [&](mesh::MeshEntityId s)
//            {
//                point_type x = m->point(s);
//
//#define DEF(_N_)  a##_N_[s]=1;   s##_N_[s]=0;                                     \
//
////        if (od[_N_] ==-1)                                                         \
////        {                                                                           \
////            Real r = (xmax[_N_] - x[_N_]) / (xmax[_N_] - xmin[_N_]);                        \
////            a##_N_[s] = alpha_(r, expN, dB);                                            \
////            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (xmax[_N_] - xmin[_N_]);     \
////        }                                                                           \
////        else if (od[_N_] ==1)                                                    \
////        {                                                                           \
////            Real r = (x[_N_] - xmin[_N_]) / (xmax[_N_] - xmin[_N_]);                        \
////            a##_N_[s] = alpha_(r, expN, dB);                                            \
////            s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (xmax[_N_] - xmin[_N_]);     \
////        };
//
//                DEF(0)
//                DEF(1)
//                DEF(2)
//#undef DEF
//            }
//    );
}

template<typename TM>
void PML<TM>::deploy()
{

    E.clear();
    B.clear();
    dX1.clear();
    dX2.clear();
    declare_global(&E, "E");
    declare_global(&B, "B");
}

template<typename TM>
void PML<TM>::sync(mesh::TransitionMap const &t_map, simulation::ProblemDomain const &other)
{


    auto E2 = static_cast<field_t<scalar_type, mesh::EDGE> const *>( other.attribute("E"));
    auto B2 = static_cast<field_t<scalar_type, mesh::FACE> const *>( other.attribute("B"));


    t_map.direct_map(mesh::EDGE,
                     [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2) { E[s1] = (*E2)[s2]; }
    );

    t_map.direct_map(mesh::FACE,
                     [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2) { B[s1] = (*B2)[s2]; }
    );

//    if (m_mesh_->name() == "PML_0")
//    {
//        t_map.direct_map(mesh::EDGE,
//                         [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2)
//                         {
//
//                             INFORM << "("
//                             << (s1.x >> 1) << ", "
//                             << (s1.y >> 1) << ", "
//                             << (s1.z >> 1) << " ) -( "
//                             << (s2.x >> 1) << ", "
//                             << (s2.y >> 1) << ", "
//                             << (s2.z >> 1) << ") "
//                             << "E =" << E[s1]
//
//                             << std::endl;
//
//
//                         }
//        );
//
//    }


}

template<typename TM>
void PML<TM>::next_step(Real dt)
{
//    VERBOSE << "next_step:\tPML  \t [Mesh Block : " << m->box() << "] " << std::endl;

    DEFINE_PHYSICAL_CONST

    B -= curl(E) * (dt * 0.5);
    E += (curl(B) * speed_of_light2) * dt;
    B -= curl(E) * (dt * 0.5);

//    dX1 = (-2.0 * dt * s0 * X10 + curl_pdx(B) / (mu0 * epsilon0) * dt) / (a0 + s0 * dt);
//    X10 += dX1;
//    E += dX1;
//
//    dX1 = (-2.0 * dt * s1 * X11 + curl_pdy(B) / (mu0 * epsilon0) * dt) / (a1 + s1 * dt);
//    X11 += dX1;
//    E += dX1;
//
//    dX1 = (-2.0 * dt * s2 * X12 + curl_pdz(B) / (mu0 * epsilon0) * dt) / (a2 + s2 * dt);
//    X12 += dX1;
//    E += dX1;
//
//
//    dX2 = (-2.0 * dt * s0 * X20 + curl_pdx(E) * dt) / (a0 + s0 * dt);
//    X20 += dX2;
//    B -= dX2;
//
//    dX2 = (-2.0 * dt * s1 * X21 + curl_pdy(E) * dt) / (a1 + s1 * dt);
//    X21 += dX2;
//    B -= dX2;
//
//    dX2 = (-2.0 * dt * s2 * X22 + curl_pdz(E) * dt) / (a2 + s2 * dt);
//    X22 += dX2;
//    B -= dX2;

}
} //namespace simpla


#endif //SIMPLA_PML_H
