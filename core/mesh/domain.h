/**
 * @file domain.h
 *
 *  Created on: 2015年4月19日
 *      Author: salmon
 */

#ifndef CORE_MESH_DOMAIN_H_
#define CORE_MESH_DOMAIN_H_

#include <stddef.h>
#include <cstdbool>
#include <iterator>
#include <memory>
#include <set>
#include <type_traits>

#include "../gtl/iterator/sp_ntuple_range.h"
#include "../gtl/ntuple.h"
#include "mesh_ids.h"

namespace simpla
{
template<typename ...> struct _Field;

template<typename TM, size_t IFORM>
struct Domain;
typedef std::integral_constant<int, 0> null_domain;
typedef std::integral_constant<int, 1> full_domain;

template<typename T>
struct domain_traits
{
	typedef full_domain type;
};

template<size_t IFORM, typename TM, typename ...Others>
struct domain_traits<_Field<Domain<TM, IFORM>, Others...>>
{
	typedef Domain<TM, IFORM> type;
};

template<typename TM, size_t IFORM>
struct Domain
{

public:
	typedef TM mesh_type;

	static constexpr size_t iform = IFORM;
	static constexpr size_t ndims = mesh_type::ndims;

	typedef Domain<mesh_type, iform> this_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::index_tuple index_tuple;
	typedef sp_nTuple_range<index_type,
			((iform == VERTEX || iform == VOLUME) ? ndims : ndims + 1)> range_type;

	template<typename TV>
	using field_value_type=typename std::conditional<(iform == VERTEX || iform == VOLUME),TV,nTuple<TV,3>>::type;

	mesh_type const &m_mesh_;
	range_type m_box_;
	std::set<id_type> m_id_set_;
public:

	Domain(mesh_type const &m)
			: m_mesh_(m)
	{
		deploy();
	}
	template<typename T0, typename T1>
	Domain(mesh_type const &m, T0 const & b, T1 const & e)
			: m_mesh_(m)
	{
		reset_bound_box(b, e);
	}

	Domain(this_type const & other)
			: m_mesh_(other.m_mesh_), m_box_(other.m_box_), m_id_set_(
					other.m_id_set_)
	{
	}
	Domain(this_type && other)
			: m_mesh_(other.m_mesh_), m_box_(other.m_box_), m_id_set_(
					other.m_id_set_)
	{
	}
	template<size_t IF>
	Domain<mesh_type, IF> clone() const
	{
		Domain<mesh_type, IF> res(m_mesh_);
		res.reset_bound_box(m_box_.m_b_, m_box_.m_e_);
		return std::move(res);
	}

	mesh_type const &
	mesh() const
	{
		return m_mesh_;
	}

	bool is_valid() const
	{
		return m_mesh_.is_valid();
	}
	bool is_simply() const
	{
		return m_id_set_.size() == 0;
	}
	bool empty() const
	{
		return is_null();
	}
	/** @name set
	 *  @{
	 */
	bool is_null() const
	{
		return is_simply() && m_box_.is_null();
	}
	bool is_full() const
	{
		return is_simply() && !is_null();
	}

	operator bool() const
	{
		return !is_null();
	}

	void clear()
	{
		m_box_.clear();
	}

	size_t size() const
	{
		if (is_simply())
		{
			return m_box_.size();
		}
		else
		{
			return m_id_set_.size();
		}
	}
	/** @} */

	this_type operator=(this_type const &other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	void swap(this_type &other)
	{
		std::swap(m_box_, other.m_box_);
		std::swap(m_mesh_, other.m_mesh_);
		std::swap(m_id_set_, other.m_id_set_);
	}
	void deploy()
	{
		reset_bound_box(m_mesh_.m_index_offset_,
				m_mesh_.m_index_offset_ + m_mesh_.m_index_count_);
	}

	std::set<id_type> & id_set()
	{
		return m_id_set_;
	}
	std::set<id_type> const& id_set() const
	{
		return m_id_set_;
	}

	size_t max_hash() const
	{
		return mesh().template max_hash<iform>();
	}
	template<typename ...Args>
	size_t hash(Args && ...args) const
	{
		return mesh().template hash<iform>(std::forward<Args>(args)...);
	}

	template<typename TFun>
	void for_each(TFun const & fun) const
	{
		if (empty())
		{
			return;
		}
		else if (is_simply())
		{

			for (auto const &idx : m_box_)
			{
				fun(m_mesh_.template pack_index<iform>(idx));
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

	template<typename TFun>
	void for_each_coordinates(TFun const & fun) const
	{
		for_each([&](id_type s)
		{
			fun(m_mesh_.coordinates(s));
		});

	}
	template<typename TFun>
	void for_each(null_domain, TFun const & fun) const
	{
	}
	template<typename TFun>
	void for_each(full_domain, TFun const & fun) const
	{

		if (is_simply())
		{
			for (auto const &x : m_box_)
			{
				fun(m_mesh_.template pack_index<iform>(x));
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
	template<typename TFun>
	void for_each(this_type const& other, TFun const & fun) const
	{
		if (is_null())
		{
			return;
		}
		else if (is_simply() && other.is_simply())
		{
			range_type r = m_box_ & other.m_box_;

			for (auto const &idx : r)
			{
				fun(mesh_type::template pack_index<iform>(idx));
			}
		}
		else if (is_simply())
		{
			for (auto const & s : other.id_set())
			{
				if (in_box(s))
				{
					fun(s);
				}
			}

		}
		else if (other.is_simply())
		{

			for (auto const & s : id_set())
			{
				if (other.in_box(s))
				{
					fun(s);
				}
			}

		}
		else
		{
			for (auto const & s : other.id_set())
			{
				if (m_id_set_.find(s) != m_id_set_.end())
				{
					fun(s);
				}
			}
		}

	}

	bool in_box(id_type s) const
	{
		return m_box_.in_bound(m_mesh_.template unpack_index<iform>(s));
	}

	void update_bound_box()
	{
		if (m_id_set_.size() > 0)
		{
			UNIMPLEMENTED2("TODO find bound of indices,"
					"and remove ids which are out of bound");
		}
		else
		{
			m_box_.m_b_ = m_box_.m_e_;
		}
	}
	void clear_ids()
	{
		if (m_id_set_.size() > 0)
		{
			UNIMPLEMENTED2("TODO: clear ids that out of bound.");
		}
	}

	template<typename T0, typename T1>
	void reset_bound_box(T0 const & b, T1 const & e)
	{

		typename range_type::value_type ib, ie;

		ib = b;
		ie = e;
		if (iform == EDGE || iform == FACE)
		{
			ib[ndims] = 0;
			ie[ndims] = 3;
		}

		range_type(ib, ie).swap(m_box_);

		clear_ids();
	}

	void reset_bound_box(coordinates_type const & b, coordinates_type const & e)
	{
		reset_bound_box(m_mesh_.coordinates_to_index(b),
				m_mesh_.coordinates_to_index(e) + 1);
	}

	template<typename T0, typename T1>
	this_type select(T0 const & b, T1 const & e) const
	{
		this_type res(*this);
		res.reset_bound_box(b, e);
		return std::move(res);
	}
	template<typename TPred>
	void filter(TPred const& pred)
	{
		std::set<id_type> res;
		if (is_simply())
		{

			std::copy_if(this->begin(), this->end(),
					std::inserter(res, res.begin()), pred);

		}
		else
		{
			std::copy_if(m_id_set_.begin(), m_id_set_.end(),
					std::inserter(res, res.begin()), pred);
		}
		res.swap(m_id_set_);
//		update_bound_box();
	}
	template<typename TPred>
	void filter_by_coordinates(TPred const& pred)
	{
		std::set<id_type> res;
		if (is_simply())
		{

			std::copy_if(this->begin(), this->end(),
					std::inserter(res, res.begin()), [&](id_type const& s)
					{	return pred(m_mesh_.coordinates(s));});

		}
		else
		{
			std::copy_if(m_id_set_.begin(), m_id_set_.end(),
					std::inserter(res, res.begin()), [&](id_type const& s)
					{	return pred(m_mesh_.coordinates(s));});
		}
		res.swap(m_id_set_);
		update_bound_box();
	}

	template<typename Tit>
	void add(Tit const & b, Tit const&e)
	{
		std::transform(b, e, std::inserter(m_id_set_, m_id_set_.begin()),
				[&](id_type const &s)
				{
					return in_bound(m_mesh_.coordinates(s));
				});
	}
	template<typename Pred>
	void remove(Pred const & pred)
	{
		filter([&](id_type const & s)
		{
			return !pred(s);
		});
	}
	template<typename Pred>
	void remove_by_coordinates(Pred const & pred)
	{
		filter([&](id_type const & s)
		{
			return !pred(m_mesh_.coordinates(s));
		});
	}
	struct iterator;
	typedef iterator const_iterator;

	const_iterator begin() const
	{
		return std::move(const_iterator(m_box_.begin()));
	}

	const_iterator end() const
	{
		return std::move(const_iterator(m_box_.end()));
	}

	struct iterator:	public std::iterator<
								typename range_type::iterator::iterator_category,
								id_type, id_type>,
						public range_type::iterator
	{
		typedef typename range_type::iterator base_iterator;

		iterator(base_iterator const &other)
				: base_iterator(other)
		{
		}

		~iterator()
		{
		}

		id_type operator*() const
		{
			return mesh_type::template pack_index<iform>(base_iterator::operator *());
		}
	};
	/**
	 * @name  Data Shape
	 * @{
	 **/

	DataSpace dataspace() const
	{
		DataSpace res = m_mesh_.template dataspace<iform>();

		if (is_simply())
		{
			typename DataSpace::index_tuple offset, count;

			auto d_shape = res.shape();

			offset = m_box_.m_b_ - d_shape.offset;

			count = m_box_.m_e_ - m_box_.m_b_;

			res.select_hyperslab(&offset[0], nullptr, &count[0], nullptr);
		}
		else
		{
			UNIMPLEMENTED;
		}

		return std::move(res);

	}

	/** @}*/

//	/** @name logical operations
//	 *  @{
//	 */
//
//	this_type const & operator &(full_domain) const
//	{
//		return *this;
//	}
//	null_domain operator &(null_domain) const
//	{
//		return null_domain();
//	}
//	this_type operator &(this_type const & other) const
//	{
//		this_type res(m_mesh_);
//
//		if (is_simply() && other.is_simply())
//		{
//			res.m_box_ = res.m_box_ & other.m_box_;
//		}
//		else if (other.is_simply())
//		{
//			for (auto const &s : m_id_set_)
//			{
//				if (other.in_box(s))
//				{
//					res.m_id_set_.insert(s);
//				}
//			}
//		}
//		else if (is_simply())
//		{
//			for (auto const &s : other.id_set())
//			{
//				if (in_box(s))
//				{
//					res.m_id_set_.insert(s);
//				}
//			}
//			res.update_bound_box();
//		}
//
//		return std::move(res);
//	}
//	this_type const & operator |(null_domain) const
//	{
//		return *this;
//	}
//
//	full_domain operator |(full_domain) const
//	{
//		return full_domain();
//	}
//
//	this_type operator |(this_type const & other) const
//	{
//		this_type res(*this);
//		return std::move(res);
//	}
//
//	this_type const & operator ^(null_domain) const
//	{
//		return *this;
//	}
//
//	full_domain operator ^(full_domain) const
//	{
//		return full_domain();
//	}
//
//	this_type operator ^(this_type const & other) const
//	{
//		this_type res(*this);
//		return std::move(res);
//	}
	/** @} */
};

namespace _impl
{

HAS_MEMBER_FUNCTION(domain)

}  // namespace _impl

template<typename T>
auto domain(T const & obj)
ENABLE_IF_DECL_RET_TYPE(
		_impl::has_member_function_domain<T>::value,obj.domain())

template<typename T>
auto domain(T const & obj)
ENABLE_IF_DECL_RET_TYPE(
		!_impl::has_member_function_domain<T>::value,full_domain())

//template<typename ...>class Domain;
//
//template<size_t NDIMS, size_t INIFIT_AXIS>
//struct Domain<TM, IFORM> : public MeshIDs_<NDIMS, INIFIT_AXIS>
//{
//	typedef Domain<MeshIDs_<NDIMS, INIFIT_AXIS> > this_type;
//
//	typedef MeshIDs_<NDIMS, INIFIT_AXIS> base_type;
//	using base_type::ndims;
//	using typename base_type::index_type;
//	using typename base_type::index_tuple;
//	using typename base_type::id_type;
//	using typename base_type::coordinates_type;
//
//	/**
//	 *
//	 *   a----------------------------b
//	 *   |                            |
//	 *   |     c--------------d       |
//	 *   |     |              |       |
//	 *   |     |  e*******f   |       |
//	 *   |     |  *       *   |       |
//	 *   |     |  *       *   |       |
//	 *   |     |  *       *   |       |
//	 *   |     |  *********   |       |
//	 *   |     ----------------       |
//	 *   ------------------------------
//	 *
//	 *   a=0
//	 *   b-a = dimension
//	 *   e-a = offset
//	 *   f-e = count
//	 *   d-c = local_dimension
//	 *   c-a = local_offset
//	 */
//
//	index_tuple m_index_dimensions_ = { 1, 1, 1 };
//	index_tuple m_index_offset_ = { 0, 0, 0 };
//	index_tuple m_index_count_ = { 1, 1, 1 };
//
//	index_tuple m_index_local_dimensions_ = { 0, 0, 0 };
//	index_tuple m_index_local_offset_ = { 0, 0, 0 };
//
//	std::set<id_type> m_id_set_;
//
//	typename base_type::template id_hasher<> m_hasher_;
//
//	Domain()
//	{
//	}
//
//	template<typename T0>
//	Domain(T0 const &d)
//	{
//		m_index_dimensions_ = d;
//		m_index_offset_ = 0;
//	}
//
//	Domain(this_type const & other) :
//			m_index_dimensions_(other.m_index_dimensions_),
//
//			m_index_offset_(other.m_index_offset_),
//
//			m_index_count_(other.m_index_count_),
//
//			m_index_local_dimensions_(other.m_index_local_dimensions_),
//
//			m_index_local_offset_(other.m_index_local_offset_),
//
//			m_id_set_(other.m_id_set_),
//
//			m_hasher_(other.m_hasher_)
//	{
//	}
//
//	void swap(this_type & other)
//	{
//		std::swap(m_index_dimensions_, other.m_index_dimensions_);
//		std::swap(m_index_offset_, other.m_index_offset_);
//		std::swap(m_index_count_, other.m_index_count_);
//
//		std::swap(m_index_local_dimensions_, other.m_index_local_dimensions_);
//		std::swap(m_index_local_offset_, other.m_index_local_offset_);
//
//		std::swap(m_id_set_, other.m_id_set_);
//		std::swap(m_hasher_, other.m_hasher_);
//	}
//
//	this_type operator=(this_type const &other)
//	{
//		this_type(other).swap(*this);
//		return *this;
//	}
//
//	template<typename OS>
//	OS & print(OS &os) const
//	{
//		os << " Dimensions =  " << m_index_dimensions_;
//
//		return os;
//
//	}
//
//	this_type const & domain() const
//	{
//		return *this;
//	}
//
//	void deploy()
//	{
//		decompose();
//
//		typename base_type::template id_hasher<>(m_index_local_dimensions_,
//				m_index_offset_ - m_index_local_offset_).swap(m_hasher_);
//	}
//	void decompose(size_t const * gw = nullptr)
//	{
//
//		CHECK(m_index_dimensions_);
//
//		m_index_count_ = m_index_dimensions_;
//		m_index_offset_ = 0;
//
//		if (GLOBAL_COMM.num_of_process() > 1)
//		{
//			GLOBAL_COMM.decompose(ndims, &m_index_count_[0],
//					&m_index_offset_[0]);
//		}
//
//		index_tuple ghost_width;
//
//		if (gw != nullptr)
//		{
//			ghost_width = gw;
//		}
//		else
//		{
//			ghost_width = 0;
//		}
//
//		m_index_local_dimensions_ = m_index_count_ + ghost_width * 2;
//
//		m_index_local_offset_ = m_index_offset_ - ghost_width;
//	}
//
//	bool is_continue() const
//	{
//		return (m_id_set_.size() == 0);
//	}
//
//	template<size_t IFORM = VERTEX>
//	typename base_type::template range_type<IFORM> range() const
//	{
//		return typename base_type::template range_type<IFORM>(m_index_offset_,
//				m_index_offset_ + m_index_count_);
//	}
//
//	std::set<id_type> &id_set() const
//	{
//		return m_id_set_;
//	}
//	template<typename TI> void dimensions(TI const & d)
//	{
//		m_index_dimensions_ = d;
//	}
//	index_tuple dimensions() const
//	{
//		return m_index_dimensions_;
//	}
//	/**
//	 * @name  Data Shape
//	 * @{
//	 **/
//
//	template<size_t IFORM = VERTEX>
//	DataSpace dataspace() const
//	{
//		nTuple<index_type, ndims + 1> f_dims;
//		nTuple<index_type, ndims + 1> f_offset;
//		nTuple<index_type, ndims + 1> f_count;
//		nTuple<index_type, ndims + 1> f_ghost_width;
//
//		int f_ndims = ndims;
//
//		f_dims = m_index_dimensions_;
//
//		f_offset = m_index_offset_;
//
//		f_count = m_index_count_;
//
//		f_ghost_width = m_index_offset_ - m_index_local_offset_;
//
//		if ((IFORM != VERTEX && IFORM != VOLUME))
//		{
//			f_ndims = ndims + 1;
//			f_dims[ndims] = 3;
//			f_offset[ndims] = 0;
//			f_count[ndims] = 3;
//			f_ghost_width[ndims] = 0;
//		}
//
//		DataSpace res(f_ndims, &(f_dims[0]));
//
//		res
//
//		.select_hyperslab(&f_offset[0], nullptr, &f_count[0], nullptr)
//
//		.convert_to_local(&f_ghost_width[0]);
//
//		return std::move(res);
//
//	}
//
//	template<size_t IFORM = VERTEX>
//	void ghost_shape(std::vector<mpi_ghosts_shape_s> *res) const
//	{
//		nTuple<size_t, ndims + 1> f_dims;
//		nTuple<size_t, ndims + 1> f_offset;
//		nTuple<size_t, ndims + 1> f_count;
//		nTuple<size_t, ndims + 1> f_ghost_width;
//		int f_ndims = ndims;
//
//		f_dims = m_index_local_dimensions_;
//
//		f_offset = m_index_offset_ - m_index_local_offset_;
//
//		f_count = m_index_count_;
//
//		f_ghost_width = f_offset;
//
//		if ((IFORM != VERTEX && IFORM != VOLUME))
//		{
//			f_ndims = ndims + 1;
//			f_dims[ndims] = 3;
//			f_offset[ndims] = 0;
//			f_count[ndims] = 3;
//			f_ghost_width[ndims] = 0;
//		}
//
//		get_ghost_shape(f_ndims, &f_dims[0], &f_offset[0], nullptr, &f_count[0],
//				nullptr, &f_ghost_width[0], res);
//
//	}
//	template<size_t IFORM = VERTEX>
//	std::vector<mpi_ghosts_shape_s> ghost_shape() const
//	{
//		std::vector<mpi_ghosts_shape_s> res;
//		ghost_shape<IFORM>(&res);
//		return std::move(res);
//	}
//	/** @}*/
//
//	auto hasher() const
//	DECL_RET_TYPE (m_hasher_)
//
//	template<size_t IFORM>
//	size_t max_hash() const
//	{
//		return m_hasher_.max_hash();
//	}
//	template<size_t IFORM, typename ...Args>
//	size_t hash(Args && ...args) const
//	{
//		return m_hasher_(std::forward<Args>(args)...);
//	}
//
//	template<size_t IFORM, typename TFun>
//	void for_each(TFun const & fun) const
//	{
//		if (m_id_set_.size() == 0)
//		{
//			auto r = range<IFORM>();
//			for (auto s : r)
//			{
//				fun(s);
//			}
//		}
//		else
//		{
//			for (auto s : m_id_set_)
//			{
//				fun(s);
//			}
//		}
//	}
//};

}// namespace simpla

#endif /* CORE_MESH_DOMAIN_H_ */
