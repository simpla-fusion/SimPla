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
#include "../../data_model/dataset.h"
#include "../../data_model/dataspace.h"
#include "../manifold_traits.h"

namespace simpla
{
/**
 * @ingroup manifold
 */
namespace manifold { namespace policy
{


template<typename TGeo>
struct StoragePolicy
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


    template<typename TV, int IFORM>
    std::shared_ptr<TV> data() const
    {
        return sp_alloc_array<TV>(memory_size<IFORM>());
    }

    template<int IFORM>
    size_t memory_size() const
    {
        return std::get<1>(dataspace<IFORM>()).size();
    }

    template<typename TV, int IFORM>
    DataSet dataset(std::shared_ptr<TV> d = nullptr) const
    {
        DataSet res;

        //FIXME  temporary by pass for XDMF

        auto dtype = traits::datatype<TV>::create();

        if (dtype.is_array() && (IFORM == VERTEX || IFORM == VOLUME))
        {
            res.datatype = dtype.element_type();
            std::tie(res.dataspace, res.memory_space) = dataspace<EDGE>();
        }
        else
        {
            res.datatype = traits::datatype<TV>::create();
            std::tie(res.dataspace, res.memory_space) = dataspace<IFORM>();

        }

        if (d != nullptr) { res.data = d; }

        return std::move(res);
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

                DataSpace(f_ndims, &(f_dims[0])).select_hyperslab(&f_start[0], nullptr, &count[0], nullptr),

                DataSpace(f_ndims, &(m_dims[0]))
                        .select_hyperslab(&m_start[0], nullptr, &count[0], nullptr)

        );

    }
};//template<typename TGeo> struct StoragePolicy
}} //namespace manifold //namespace policy



}//namespace simpla
#endif //SIMPLA_STORAGE_POLICY_H
