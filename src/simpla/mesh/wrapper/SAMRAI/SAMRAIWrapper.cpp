//
// Created by salmon on 16-10-24.
//
#include <simpla/toolbox/nTuple.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Patch.h>
#include <simpla/simulation/Context.h>

#include <SAMRAI/hier/VariableDatabase.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/hier/BaseGridGeometry.h>
#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/pdat/CellData.h>
#include <SAMRAI/pdat/EdgeData.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/NodeData.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/pdat/EdgeVariable.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/pdat/NodeVariable.h>


namespace simpla
{

namespace detail
{

template<typename V, mesh::MeshEntityType IFORM> struct SAMRAITraitsPatch;
template<typename V> struct SAMRAITraitsPatch<V, mesh::VERTEX> { typedef SAMRAI::pdat::NodeData<V> type; };
template<typename V> struct SAMRAITraitsPatch<V, mesh::EDGE> { typedef SAMRAI::pdat::EdgeData<V> type; };
template<typename V> struct SAMRAITraitsPatch<V, mesh::FACE> { typedef SAMRAI::pdat::FaceData<V> type; };
template<typename V> struct SAMRAITraitsPatch<V, mesh::VOLUME> { typedef SAMRAI::pdat::CellData<V> type; };

template<typename T>
SAMRAI::hier::Index samraiIndexConvert(nTuple<T, 2> const &v) { return SAMRAI::hier::Index(v[0], v[1]); }

template<typename T>
SAMRAI::hier::Index samraiIndexConvert(nTuple<T, 3> const &v) { return SAMRAI::hier::Index(v[0], v[1], v[2]); }


template<typename T>
SAMRAI::hier::IntVector samraiIntVectorConvert(nTuple<T, 2> const &v)
{
    int d[2] = {v[0], v[1]};

    return SAMRAI::hier::IntVector(SAMRAI::tbox::Dimension(2), d);
}

template<typename T>
SAMRAI::hier::IntVector samraiIntVectorConvert(nTuple<T, 3> const &v)
{
    int d[2] = {v[0], v[1], v[2]};

    return SAMRAI::hier::IntVector(SAMRAI::tbox::Dimension(3), d);
}

template<typename V, typename M, mesh::MeshEntityType IFORM>
class SAMRAIWrapperPatch
        : public SAMRAITraitsPatch<V, IFORM>::type,
          public mesh::Patch<V, M, IFORM>
{
    typedef typename SAMRAITraitsPatch<V, IFORM>::type samari_base_type;
    typedef mesh::Patch<V, M, IFORM> simpla_base_type;
public:
    SAMRAIWrapperPatch(std::shared_ptr<M> const &m, size_tuple const &gw)
            : samari_base_type(SAMRAI::hier::Box(samraiIndexConvert(std::get<0>(m->index_box())),
                                                 samraiIndexConvert(std::get<1>(m->index_box())),
                                                 SAMRAI::hier::BlockId(0)),
                               1, samraiIntVectorConvert(gw)),
              simpla_base_type(m->get()) {}

    ~SAMRAIWrapperPatch() {}
};


template<typename V, mesh::MeshEntityType IFORM> struct SAMRAITraitsVariable;
template<typename V> struct SAMRAITraitsVariable<V, mesh::VERTEX> { typedef SAMRAI::pdat::NodeVariable<V> type; };
template<typename V> struct SAMRAITraitsVariable<V, mesh::EDGE> { typedef SAMRAI::pdat::EdgeVariable<V> type; };
template<typename V> struct SAMRAITraitsVariable<V, mesh::FACE> { typedef SAMRAI::pdat::FaceVariable<V> type; };
template<typename V> struct SAMRAITraitsVariable<V, mesh::VOLUME> { typedef SAMRAI::pdat::CellVariable<V> type; };

template<typename V, typename M, mesh::MeshEntityType IFORM>
class SAMRAIWrapperAttribute
        : public SAMRAITraitsVariable<V, IFORM>::type,
          public mesh::Attribute<SAMRAIWrapperPatch<V, M, IFORM> >
{
    typedef typename SAMRAITraitsVariable<V, IFORM>::type samrai_base_type;
    typedef mesh::Attribute<SAMRAIWrapperPatch<V, M, IFORM>> simpla_base_type;
public:
    template<typename TM>
    SAMRAIWrapperAttribute(std::shared_ptr<TM> const &m, std::string const &name) :
            samrai_base_type(SAMRAI::tbox::Dimension(M::ndims), name, 1), simpla_base_type(m) {}

    ~SAMRAIWrapperAttribute() {}
};


class SAMRAIWrapperAtlas
        : public mesh::Atlas,
          public SAMRAI::hier::PatchHierarchy
{

    typedef mesh::Atlas simpla_base_type;
    typedef SAMRAI::hier::PatchHierarchy samrai_base_type;
public:
    SAMRAIWrapperAtlas(std::string const &name)
            : samrai_base_type(name,
                               boost::shared_ptr<SAMRAI::hier::BaseGridGeometry>(
                                       new SAMRAI::geom::CartesianGridGeometry(
                                               SAMRAI::tbox::Dimension(3),
                                               "CartesianGridGeometry",
                                               boost::shared_ptr<SAMRAI::tbox::Database>(nullptr)))
    )
    {

    }

    ~SAMRAIWrapperAtlas() {}
};

std::shared_ptr<mesh::AttributeBase>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
                      std::shared_ptr<mesh::Atlas> const &m, std::string const &name)
{
}


std::shared_ptr<mesh::AttributeBase>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
                      std::shared_ptr<mesh::MeshBase> const &m, std::string const &name)
{
}

std::shared_ptr<mesh::PatchBase>
create_patch_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
                  std::shared_ptr<mesh::MeshBase> const &m)
{

}
}//namespace detail


struct SAMRAIWrapperContext : public simulation::ContextBase
{
    void setup() {};

    void teardown() {};

    std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

    toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const { return os; };

    toolbox::IOStream &load(toolbox::IOStream &is) { return is; };

    toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb) {};

    std::shared_ptr<mesh::DomainBase> get_domain(uuid id) const {};

    void sync(int level = 0, int flag = 0) {};

    void run(Real dt, int level = 0) {};

    Real time() const {};

    void time(Real t) {};

    void next_time_step(Real dt) {};


private:
    std::shared_ptr<detail::SAMRAIWrapperAtlas> m_atlas_;

};

std::shared_ptr<simulation::ContextBase> create_context(std::string const &name)
{
    return std::make_shared<SAMRAIWrapperContext>();
}

} //namespace simpla