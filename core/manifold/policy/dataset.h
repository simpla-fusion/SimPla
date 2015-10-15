/**
 * @file dataset.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_DATASET_H
#define SIMPLA_DATASET_H

#include "../../gtl/dataset/dataspace.h"
#include "../manifold_traits.h"

namespace simpla
{
template<typename ...> struct DataSetPolicy;

template<typename TGeo>
struct DataSetPolicy<TGeo>
{
private:

	typedef TGeo geometry_type;


	typedef DataSetPolicy<geometry_type> this_type;

	geometry_type const &m_geo_;


public:
	DataSetPolicy(geometry_type &geo) : m_geo_(geo) { }

	virtual ~DataSetPolicy() { }


	template<size_t IFORM>
	DataSpace dataspace() const
	{
		typedef typename geometry_type::index_type index_type;

		static constexpr int ndims = geometry_type::ndims;

		nTuple<index_type, ndims + 1> f_dims;
		nTuple<index_type, ndims + 1> f_offset;
		nTuple<index_type, ndims + 1> f_count;
		nTuple<index_type, ndims + 1> f_ghost_width;

		nTuple<index_type, ndims + 1> m_dims;
		nTuple<index_type, ndims + 1> m_offset;

		int f_ndims = ndims;

		f_dims = geometry_type::unpack_index(m_geo_.m_id_max_ - m_geo_.m_id_min_);

		f_offset = geometry_type::unpack_index(m_geo_.m_id_local_min_ - m_geo_.m_id_min_);

		f_count = geometry_type::unpack_index(
				m_geo_.m_id_local_max_ - m_geo_.m_id_local_min_);

		m_dims = geometry_type::unpack_index(
				m_geo_.m_id_memory_max_ - m_geo_.m_id_memory_min_);;

		m_offset = geometry_type::unpack_index(m_geo_.m_id_local_min_ - m_geo_.m_id_min_);

		if ((IFORM == EDGE || IFORM == FACE))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
			m_dims[ndims] = 3;
			m_offset[ndims] = 0;
		}
		else
		{
			f_ndims = ndims;
			f_dims[ndims] = 1;
			f_offset[ndims] = 0;
			f_count[ndims] = 1;
			m_dims[ndims] = 1;
			m_offset[ndims] = 0;
		}

		DataSpace res(f_ndims, &(f_dims[0]));

		res.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)
				.set_local_shape(&m_dims[0], &m_offset[0]);

		return std::move(res);

	}
};//template<typename TGeo> struct DataSetPolicy

}//namespace simpla
#endif //SIMPLA_DATASET_H
