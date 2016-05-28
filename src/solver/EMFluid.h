/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include "../field/Field.h"

#include "../mesh/Mesh.h"
#include "../mesh/MeshWorker.h"
#include "../physics/PhysicalConstants.h"
#include "../mesh/MeshEntityIterator.h"
#include "../manifold/pre_define/PreDefine.h"

namespace simpla { namespace phy_solver
{
using namespace simpla::mesh;


template<typename TM>
class EMFluid : public mesh::MeshWorker
{
    typedef mesh::MeshWorker base_type;
    typedef EMFluid<TM> this_type;

public:
    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;

    mesh::MeshAtlas const *m_atlas_;

    mesh_type const *m;

    EMFluid(mesh::MeshAtlas const &m_p) : m_atlas_(&m_p) { }

    EMFluid(mesh_type const &m_p) : m_atlas_(nullptr), m(&m_p) { }

    virtual   ~EMFluid() { }


    virtual void next_step(Real dt);

    virtual void tear_down();

    virtual io::IOStream &check_point(io::IOStream &) const;


    mesh::MeshEntityRange limiter_boundary;
    mesh::MeshEntityRange vertex_boundary;
    mesh::MeshEntityRange edge_boundary;
    mesh::MeshEntityRange face_boundary;

    mesh::MeshEntityRange plasma_region_volume;
    mesh::MeshEntityRange plasma_region_vertex;
    mesh::MeshEntityRange J_src;

    template<typename ValueType, size_t IFORM> using field_t =  Field<ValueType, TM, std::integral_constant<size_t, IFORM> >;;

//    std::function< Vec3(Real, point_type const &)
//
//    >
//    J_src_fun;


    typedef field_t<scalar_type, FACE> TB;
    typedef field_t<scalar_type, EDGE> TE;
    typedef field_t<scalar_type, EDGE> TJ;
    typedef field_t<scalar_type, VERTEX> TRho;
    typedef field_t<vector_type, VERTEX> TJv;

    field_t<scalar_type, EDGE> E0{*m_atlas_, "E0"};
    field_t<scalar_type, FACE> B0{*m_atlas_, "B0"};
    field_t<vector_type, VERTEX> B0v{*m_atlas_, "B0v"};
    field_t<scalar_type, VERTEX> BB{*m_atlas_, "BB"};

    field_t<vector_type, VERTEX> Ev{*m_atlas_, "Ev"};
    field_t<vector_type, VERTEX> Bv{*m_atlas_, "Bv"};

    field_t<scalar_type, FACE> B1{*m_atlas_, "B1"};
    field_t<scalar_type, EDGE> E1{*m_atlas_, "E1"};
    field_t<scalar_type, EDGE> J1{*m_atlas_, "J1"};

    field_t<scalar_type, VERTEX> rho0{*m_atlas_};


    struct fluid_s
    {
        Real mass;
        Real charge;
        field_t<scalar_type, VERTEX> rho1;
        field_t<vector_type, VERTEX> J1;
    };

    std::map<std::string, fluid_s> fluid_sp;

    std::pair<typename std::map<std::string, fluid_s>::iterator, bool>
    add_particle(std::string const &name, Real mass, Real charge)
    {
        return fluid_sp.emplace(
                std::make_pair(name, fluid_s{mass, charge, TRho{*m_atlas_, "n_" + name}, TJv{*m_atlas_, "J_" + name}}));

    }
};


template<typename TM>
void EMFluid<TM>::tear_down()
{
//    out_stream.close_grid();
//
//    out_stream.open_grid("dump", io::XDMFStream::UNIFORM);
//
//    out_stream.reference_topology_geometry("Main");
//
//    out_stream.time(m.time());
//
//    for (auto const &item:m.attributes())
//    {
//        if (!item.second.lock()->properties()["DisableXDMFOutput"])
//        {
//            out_stream.write(item.first, *std::dynamic_pointer_cast<base::AttributeObject>(item.second.lock()));
//
//        } else
//        {
//            out_stream.hdf5().write(item.first, item.second.lock()->dump(), io::SP_RECORD);
//        }
//    }
//    out_stream.close_grid();
//
//    out_stream.close();
}

template<typename TM>
io::IOStream &EMFluid<TM>::check_point(io::IOStream &) const
{


}

template<typename TM>
void EMFluid<TM>::next_step(Real dt)
{

    DEFINE_PHYSICAL_CONST


    LOGGER << " Time = [" << time() << "] Count = [" << step_count() << "]" << std::endl;


    J1.clear();

//    J1.accept(J_src, [&](id_type s, Real &v) { J1.add(s, J_src_fun(time(), m->point(s))); });


    LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

    B1.accept(face_boundary, [&](id_type, Real &v) { v = 0; });


    LOG_CMD(E1 += (curl(B1) * speed_of_light2 - J1 / epsilon0) * dt);

    E1.accept(edge_boundary, [&](id_type, Real &v) { v = 0; });


    field_t<vector_type, VERTEX> dE{*m_atlas_};



    //particle::absorb(ion, limiter_boundary);
    if (fluid_sp.size() > 0)
    {

        field_t<vector_type, VERTEX> Q{*m_atlas_};
        field_t<vector_type, VERTEX> K{*m_atlas_};

        field_t<scalar_type, VERTEX> a{*m_atlas_};
        field_t<scalar_type, VERTEX> b{*m_atlas_};
        field_t<scalar_type, VERTEX> c{*m_atlas_};

        a.clear();
        b.clear();
        c.clear();

        Q = map_to<VERTEX>(E1) - Ev;


        for (auto &p :   fluid_sp)
        {

            Real ms = p.second.mass;
            Real qs = p.second.charge;


            field_t<scalar_type, VERTEX> &ns = p.second.rho1;

            field_t<vector_type, VERTEX> &Js = p.second.J1;;


            Real as = (dt * qs) / (2.0 * ms);

            Q -= 0.5 * dt / epsilon0 * Js;

            K = (Ev * qs * ns * 2.0 + cross(Js, B0v)) * as + Js;

            Js = (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);

            Q -= 0.5 * dt / epsilon0 * Js;

            a += qs * ns * (as / (BB * as * as + 1));
            b += qs * ns * (as * as / (BB * as * as + 1));
            c += qs * ns * (as * as * as / (BB * as * as + 1));


        }

        a *= 0.5 * dt / epsilon0;
        b *= 0.5 * dt / epsilon0;
        c *= 0.5 * dt / epsilon0;
        a += 1;


        LOG_CMD(dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) /
                     (b * b * BB + a * a));

        for (auto &p :   fluid_sp)
        {
            Real ms = p.second.mass;
            Real qs = p.second.charge;
            field_t<scalar_type, VERTEX> &ns = p.second.rho1;
            field_t<vector_type, VERTEX> &Js = p.second.J1;;


            Real as = (dt * qs) / (2.0 * ms);

            K = dE * ns * qs * as;
            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
        }
        Ev += dE;

//        LOG_CMD(E1 += map_to<EDGE>(Ev) - E1);
    }


    LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

    B1.accept(face_boundary, [&](id_type const &, Real &v) { v = 0; });

    base_type::next_step(dt);
}


}}//namespace simpla { namespace solver {
#endif //SIMPLA_EM_FLUID_H
