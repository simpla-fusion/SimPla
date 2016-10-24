//
// Created by salmon on 16-10-24.
//
#include <simpla/toolbox/nTuple.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Patch.h>

#include <SAMRAI/hier/VariableDatabase.h>
#include <SAMRAI/pdat/CellData.h>
#include <SAMRAI/pdat/EdgeData.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/NodeData.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/pdat/EdgeVariable.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/pdat/NodeVariable.h>


namespace simpla { namespace mesh
{
namespace SAMRAIWrapper
{

template<typename V, MeshEntityType IFORM> struct SAMRAITraitsPatch;
template<typename V> struct SAMRAITraitsPatch<V, VERTEX> { typedef SAMRAI::pdat::NodeData<V> type; };
template<typename V> struct SAMRAITraitsPatch<V, EDGE> { typedef SAMRAI::pdat::EdgeData<V> type; };
template<typename V> struct SAMRAITraitsPatch<V, FACE> { typedef SAMRAI::pdat::FaceData<V> type; };
template<typename V> struct SAMRAITraitsPatch<V, VOLUME> { typedef SAMRAI::pdat::CellData<V> type; };

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

template<typename V, typename M, MeshEntityType IFORM>
class SAMRAIWrapperPatch
        : public SAMRAITraitsPatch<V, IFORM>::type,
          public Patch<V, M, IFORM>
{
    typedef typename SAMRAITraitsPatch<V, IFORM>::type samari_base_type;
    typedef Patch<V, M, IFORM> simpla_base_type;

    SAMRAIWrapperPatch(std::shared_ptr<M> const &m, size_tuple const &gw)
            : samari_base_type(SAMRAI::hier::Box(samraiIndexConvert(std::get<0>(m->index_box())),
                                                 samraiIndexConvert(std::get<1>(m->index_box())),
                                                 SAMRAI::hier::BlockId(0)),
                               1, samraiIntVectorConvert(gw)),
              simpla_base_type(m->get()) {}

    ~SAMRAIWrapperPatch() {}
};


template<typename V, MeshEntityType IFORM> struct SAMRAITraitsVariable;
template<typename V> struct SAMRAITraitsVariable<V, VERTEX> { typedef SAMRAI::pdat::NodeVariable<V> type; };
template<typename V> struct SAMRAITraitsVariable<V, EDGE> { typedef SAMRAI::pdat::EdgeVariable<V> type; };
template<typename V> struct SAMRAITraitsVariable<V, FACE> { typedef SAMRAI::pdat::FaceVariable<V> type; };
template<typename V> struct SAMRAITraitsVariable<V, VOLUME> { typedef SAMRAI::pdat::CellVariable<V> type; };

template<typename V, typename M, MeshEntityType IFORM>
class SAMRAIWrapperAttribute
        : public SAMRAITraitsVariable<V, IFORM>::type,
          public Attribute<SAMRAIWrapperPatch<V, M, IFORM> >
{
    typedef typename SAMRAITraitsVariable<V, IFORM>::type samrai_base_type;
    typedef Attribute<SAMRAIWrapperPatch<V, M, IFORM> > simpla_base_type;

    template<typename TM>
    SAMRAIWrapperAttribute(std::shared_ptr<TM> const &m, std::string const &name) :
            samrai_base_type(SAMRAI::tbox::Dimension(M::ndims), name, 1), simpla_base_type(m) {}

    ~SAMRAIWrapperAttribute() {}
};

std::shared_ptr<AttributeBase>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, MeshEntityType const &,
                      std::shared_ptr<Atlas> const &m, std::string const &name)
{
}


std::shared_ptr<AttributeBase>
create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, MeshEntityType const &,
                      std::shared_ptr<MeshBase> const &m, std::string const &name)
{
}

std::shared_ptr<PatchBase>
create_patch_impl(std::type_info const &type_info, std::type_info const &mesh_info, MeshEntityType const &,
                  std::shared_ptr<MeshBase> const &m)
{

}
}
}}//namespace simpla{namespace mesh{