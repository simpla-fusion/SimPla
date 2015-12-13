/**
 * @file storage_policy.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_STORAGE_POLICY_H
#define SIMPLA_STORAGE_POLICY_H

#include <string.h>
#include "../../gtl/design_pattern/singleton_holder.h"
#include "../../gtl/utilities/memory_pool.h"
#include "../../dataset/dataset.h"
#include "../../dataset/dataspace.h"
#include "../manifold_traits.h"

namespace simpla
{
/**
 * @ingroup manifold
 */
namespace manifold { namespace policy
{

template<typename ...> struct StoragePolicy;

template<typename TGeo>
struct StoragePolicy<TGeo>
{
private:

    typedef TGeo geometry_type;

    typedef StoragePolicy<geometry_type> this_type;


    typedef typename TGeo::id_type id_type;

    geometry_type const &m_geo_;


public:

    typedef this_type storage_policy;


    StoragePolicy(geometry_type &geo) : m_geo_(geo) { }

    virtual ~StoragePolicy() { }

    template<typename TDict> void load(TDict const &) { }

    template<typename OS> OS &print(OS &os) const
    {
        os << "\t StoragePolicy={ Default }," << std::endl;
        return os;
    }


    void deploy() { }

    template<typename TV, typename ...Others>
    inline TV &at(DataSet &d, id_type s) const
    {
        return d.template get_value<TV>(m_geo_.hash(s));
    }


    template<typename TV, typename ...Others>
    inline TV const &at(DataSet const &d, id_type s) const
    {
        return d.template get_value<TV>(m_geo_.hash(s));
    }

    template<typename TV, int IFORM>
    std::shared_ptr<DataSet> dataset(std::shared_ptr<void> d = nullptr) const
    {
        auto res = std::make_shared<DataSet>();

        //FIXME  temporary by pass for XDMF

        auto dtype = traits::datatype<TV>::create();

        if (dtype.is_array() && (IFORM == VERTEX || IFORM == VOLUME))
        {
            res->datatype = dtype.element_type();
            std::tie(res->dataspace, res->memory_space) = dataspace<EDGE>();
        }
        else
        {
            res->datatype = traits::datatype<TV>::create();
            std::tie(res->dataspace, res->memory_space) = dataspace<IFORM>();

        }

        if (d != nullptr) { res->data = d; }

        return res;
    };


    template<int IFORM>
    std::tuple<DataSpace, DataSpace> dataspace() const
    {
        return dataspace<IFORM>(m_geo_.template range<IFORM>());
    }

    template<int IFORM>
    std::tuple<DataSpace, DataSpace> dataspace(typename geometry_type::range_type const &r) const
    {

        static constexpr int ndims = geometry_type::ndims;

        nTuple<size_t, ndims + 1> count;

        nTuple<size_t, ndims + 1> f_dims;
        nTuple<size_t, ndims + 1> f_start;

        nTuple<size_t, ndims + 1> f_ghost_width;

        nTuple<size_t, ndims + 1> m_dims;
        nTuple<size_t, ndims + 1> m_start;

        int f_ndims = ndims;

        count = (m_geo_.m_idx_local_max_ - m_geo_.m_idx_local_min_);

        f_dims = (m_geo_.m_idx_max_ - m_geo_.m_idx_min_);

        f_start = (m_geo_.m_idx_local_min_ - m_geo_.m_idx_min_);

        m_dims = (m_geo_.m_idx_memory_max_ - m_geo_.m_idx_memory_min_);

        m_start = (m_geo_.m_idx_local_min_ - m_geo_.m_idx_memory_min_);

        if ((IFORM == EDGE || IFORM == FACE))
        {
            f_ndims = ndims + 1;

            count[ndims] = 3;

            f_dims[ndims] = 3;
            f_start[ndims] = 0;

            m_dims[ndims] = 3;
            m_start[ndims] = 0;
        }
        else
        {
            f_ndims = ndims;
            count[ndims] = 1;

            f_dims[ndims] = 1;
            f_start[ndims] = 0;

            m_dims[ndims] = 1;
            m_start[ndims] = 0;
        }

        return std::make_tuple(

                DataSpace(f_ndims, &(f_dims[0]))
                        .select_hyperslab(&f_start[0], nullptr, &count[0], nullptr),

                DataSpace(f_ndims, &(m_dims[0]))
                        .select_hyperslab(&m_start[0], nullptr, &count[0], nullptr)

        );


    }
};//template<typename TGeo> struct StoragePolicy

}} //namespace manifold //namespace policy

namespace traits
{

template<typename TGeo>
struct type_id<manifold::policy::StoragePolicy<TGeo>>
{
    static std::string name()
    {
        return "StoragePolicy<" + type_id<TGeo>::name() + ">";
    }
};
}

}//namespace simpla
#endif //SIMPLA_STORAGE_POLICY_H
