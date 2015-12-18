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
#include "../../data_model/DataSet.h"
#include "../../data_model/DataSpace.h"
#include "../ManifoldTraits.h"

namespace simpla
{
/**
 * @ingroup Manifold
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

    virtual std::ostream &print(std::ostream &os) const
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
        return std::get<1>(data_space<IFORM>()).size();
    }

    template<typename TV, int IFORM>
    data_model::DataSet data_set(std::shared_ptr<TV> d = nullptr) const
    {
        data_model::DataSet res;

        //FIXME  temporary by pass for XDMF

        auto dtype = data_model::DataType::create<TV>();

        if (dtype.is_array() && (IFORM == VERTEX || IFORM == VOLUME))
        {
            res.data_type = dtype.element_type();
            std::tie(res.data_space, res.memory_space) = data_space<EDGE>();
        }
        else
        {
            res.data_type = data_model::DataType::create<TV>();
            std::tie(res.data_space, res.memory_space) = data_space<IFORM>();

        }

        if (d != nullptr) { res.data = d; }

        return std::move(res);
    };


    template<int IFORM>
    std::tuple<data_model::DataSpace, data_model::DataSpace> data_space() const
    {
        return data_space<IFORM>(m_geo_.template range<IFORM>());
    }

    template<int IFORM>
    std::tuple<data_model::DataSpace, data_model::DataSpace> data_space(
            typename geometry_type::range_type const &r) const
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

                data_model::DataSpace(f_ndims, &(f_dims[0])).select_hyperslab(&f_start[0], nullptr, &count[0], nullptr),

                data_model::DataSpace(f_ndims, &(m_dims[0]))
                        .select_hyperslab(&m_start[0], nullptr, &count[0], nullptr)

        );

    }
};//template<typename TGeo> struct StoragePolicy
}} //namespace Manifold //namespace policy



}//namespace simpla
#endif //SIMPLA_STORAGE_POLICY_H
