/*
 * domain.h
 *
 *  Created on: 2015年4月19日
 *      Author: salmon
 */

#ifndef CORE_MESH_DOMAIN_H_
#define CORE_MESH_DOMAIN_H_

namespace simpla
{

template<typename ...>class Domain;

template<size_t NDIMS, size_t INIFIT_AXIS>
struct Domain<TM, IFORM> : public MeshIDs_<NDIMS, INIFIT_AXIS>
{
	typedef Domain<MeshIDs_<NDIMS, INIFIT_AXIS> > this_type;

	typedef MeshIDs_<NDIMS, INIFIT_AXIS> base_type;
	using base_type::ndims;
	using typename base_type::index_type;
	using typename base_type::index_tuple;
	using typename base_type::id_type;
	using typename base_type::coordinates_type;

	/**
	 *
	 *   a----------------------------b
	 *   |                            |
	 *   |     c--------------d       |
	 *   |     |              |       |
	 *   |     |  e*******f   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *       *   |       |
	 *   |     |  *********   |       |
	 *   |     ----------------       |
	 *   ------------------------------
	 *
	 *   a=0
	 *   b-a = dimension
	 *   e-a = offset
	 *   f-e = count
	 *   d-c = local_dimension
	 *   c-a = local_offset
	 */

	index_tuple m_index_dimensions_ = { 1, 1, 1 };
	index_tuple m_index_offset_ = { 0, 0, 0 };
	index_tuple m_index_count_ = { 1, 1, 1 };

	index_tuple m_index_local_dimensions_ = { 0, 0, 0 };
	index_tuple m_index_local_offset_ = { 0, 0, 0 };

	std::set<id_type> m_id_set_;

	typename base_type::template id_hasher<> m_hasher_;

	Domain()
	{
	}

	template<typename T0>
	Domain(T0 const &d)
	{
		m_index_dimensions_ = d;
		m_index_offset_ = 0;
	}

	Domain(this_type const & other) :
			m_index_dimensions_(other.m_index_dimensions_),

			m_index_offset_(other.m_index_offset_),

			m_index_count_(other.m_index_count_),

			m_index_local_dimensions_(other.m_index_local_dimensions_),

			m_index_local_offset_(other.m_index_local_offset_),

			m_id_set_(other.m_id_set_),

			m_hasher_(other.m_hasher_)
	{
	}

	void swap(this_type & other)
	{
		std::swap(m_index_dimensions_, other.m_index_dimensions_);
		std::swap(m_index_offset_, other.m_index_offset_);
		std::swap(m_index_count_, other.m_index_count_);

		std::swap(m_index_local_dimensions_, other.m_index_local_dimensions_);
		std::swap(m_index_local_offset_, other.m_index_local_offset_);

		std::swap(m_id_set_, other.m_id_set_);
		std::swap(m_hasher_, other.m_hasher_);
	}

	this_type operator=(this_type const &other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	template<typename OS>
	OS & print(OS &os) const
	{
		os << " Dimensions =  " << m_index_dimensions_;

		return os;

	}

	this_type const & domain() const
	{
		return *this;
	}

	void deploy()
	{
		decompose();

		typename base_type::template id_hasher<>(m_index_local_dimensions_,
				m_index_offset_ - m_index_local_offset_).swap(m_hasher_);
	}
	void decompose(size_t const * gw = nullptr)
	{

		CHECK(m_index_dimensions_);

		m_index_count_ = m_index_dimensions_;
		m_index_offset_ = 0;

		if (GLOBAL_COMM.num_of_process() > 1)
		{
			GLOBAL_COMM.decompose(ndims, &m_index_count_[0],
					&m_index_offset_[0]);
		}

		index_tuple ghost_width;

		if (gw != nullptr)
		{
			ghost_width = gw;
		}
		else
		{
			ghost_width = 0;
		}

		m_index_local_dimensions_ = m_index_count_ + ghost_width * 2;

		m_index_local_offset_ = m_index_offset_ - ghost_width;
	}

	bool is_continue() const
	{
		return (m_id_set_.size() == 0);
	}

	template<size_t IFORM = VERTEX>
	typename base_type::template range_type<IFORM> range() const
	{
		return typename base_type::template range_type<IFORM>(m_index_offset_,
				m_index_offset_ + m_index_count_);
	}

	std::set<id_type> &id_set() const
	{
		return m_id_set_;
	}
	template<typename TI> void dimensions(TI const & d)
	{
		m_index_dimensions_ = d;
	}
	index_tuple dimensions() const
	{
		return m_index_dimensions_;
	}
	/**
	 * @name  Data Shape
	 * @{
	 **/

	template<size_t IFORM = VERTEX>
	DataSpace dataspace() const
	{
		nTuple<index_type, ndims + 1> f_dims;
		nTuple<index_type, ndims + 1> f_offset;
		nTuple<index_type, ndims + 1> f_count;
		nTuple<index_type, ndims + 1> f_ghost_width;

		int f_ndims = ndims;

		f_dims = m_index_dimensions_;

		f_offset = m_index_offset_;

		f_count = m_index_count_;

		f_ghost_width = m_index_offset_ - m_index_local_offset_;

		if ((IFORM != VERTEX && IFORM != VOLUME))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
			f_ghost_width[ndims] = 0;
		}

		DataSpace res(f_ndims, &(f_dims[0]));

		res

		.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)

		.convert_to_local(&f_ghost_width[0]);

		return std::move(res);

	}

	template<size_t IFORM = VERTEX>
	void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
	{
		nTuple<size_t, ndims + 1> f_dims;
		nTuple<size_t, ndims + 1> f_offset;
		nTuple<size_t, ndims + 1> f_count;
		nTuple<size_t, ndims + 1> f_ghost_width;
		int f_ndims = ndims;

		f_dims = m_index_local_dimensions_;

		f_offset = m_index_offset_ - m_index_local_offset_;

		f_count = m_index_count_;

		f_ghost_width = f_offset;

		if ((IFORM != VERTEX && IFORM != VOLUME))
		{
			f_ndims = ndims + 1;
			f_dims[ndims] = 3;
			f_offset[ndims] = 0;
			f_count[ndims] = 3;
			f_ghost_width[ndims] = 0;
		}

		get_ghost_shape(f_ndims, &f_dims[0], &f_offset[0], nullptr, &f_count[0],
				nullptr, &f_ghost_width[0], res);

	}
	template<size_t IFORM = VERTEX>
	std::vector<mpi_ghosts_shape_s> ghost_shape() const
	{
		std::vector<mpi_ghosts_shape_s> res;
		ghost_shape<IFORM>(&res);
		return std::move(res);
	}
	/** @}*/

	auto hasher() const
	DECL_RET_TYPE (m_hasher_)

	template<size_t IFORM>
	size_t max_hash() const
	{
		return m_hasher_.max_hash();
	}
	template<size_t IFORM, typename ...Args>
	size_t hash(Args && ...args) const
	{
		return m_hasher_(std::forward<Args>(args)...);
	}

	template<size_t IFORM, typename TFun>
	void for_each(TFun const & fun) const
	{
		if (m_id_set_.size() == 0)
		{
			auto r = range<IFORM>();
			for (auto s : r)
			{
				fun(s);
			}
		}
		else
		{
			for (auto s : m_id_set_)
			{
				fun(s);
			}
		}
	}
};

}  // namespace simpla

#endif /* CORE_MESH_DOMAIN_H_ */
