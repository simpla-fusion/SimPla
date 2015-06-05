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
struct Domain: public TM::range_type
{

public:
	typedef TM mesh_type;

	static constexpr size_t iform = IFORM;
	static constexpr size_t ndims = mesh_type::ndims;

	typedef Domain<mesh_type, iform> this_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::point_type point_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::index_tuple index_tuple;

	typedef typename mesh_type::range_type range_type;

	typedef typename range_type::iterator const_iterator;
	using range_type::begin;
	using range_type::end;

	template<typename TV>
	using field_value_type=typename std::conditional<(iform == VERTEX || iform == VOLUME),TV,nTuple<TV,3>>::type;

	mesh_type const &m_mesh_;
	std::set<id_type> m_id_set_;
public:

	Domain(mesh_type const &m)
			: range_type(m.range(mesh_type::template sub_index_to_id<iform>())), m_mesh_(
					m)
	{
	}

	Domain(this_type const & other)
			: range_type(other), m_mesh_(other.m_mesh_), m_id_set_(
					other.m_id_set_)
	{
	}
	Domain(this_type && other)
			: range_type(other), m_mesh_(other.m_mesh_), m_id_set_(
					other.m_id_set_)
	{
	}

	mesh_type const & mesh() const
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
		return is_simply() && range_type::empty();
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
		range_type::clear();
	}

	size_t size() const
	{
		if (is_simply())
		{
			return range_type::size();
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
		range_type::swap(other);
		std::swap(m_mesh_, other.m_mesh_);
		std::swap(m_id_set_, other.m_id_set_);
	}
	void deploy()
	{
		range_type(m_mesh_.template range<iform>()).swap(*this);
	}

	std::set<id_type> & id_set()
	{
		return m_id_set_;
	}
	std::set<id_type> const& id_set() const
	{
		return m_id_set_;
	}

	constexpr size_t max_hash() const
	{
		return mesh().template max_hash<iform>();
	}

	constexpr size_t hash(id_type s) const
	{
		return m_mesh_.hash(s);
	}

	constexpr id_type hash(index_type i, index_type j, index_type k,
			index_type n = 0) const
	{
		return m_mesh_.hash(m_mesh_.template pack_index<iform>(i, j, k, n));
	}

	constexpr id_type pack_relative_index(index_type i, index_type j,
			index_type k, index_type n = 0) const
	{
		return m_mesh_.template pack_relative_index<iform>(i, j, k, n);
	}
	template<typename TI>
	constexpr id_type pack_relative_index(nTuple<TI, 3> const &i, index_type n =
			0) const
	{
		return m_mesh_.template pack_relative_index<iform>(i[0], i[1], i[2], n);
	}
	template<typename TI>
	constexpr id_type pack_relative_index(nTuple<TI, 4> const & i) const
	{
		return m_mesh_.template pack_relative_index<iform>(i[0], i[1], i[2],
				i[3]);
	}

	constexpr auto unpack_relative_index(id_type s) const
	DECL_RET_TYPE((m_mesh_.unpack_relative_index(s)))

	template<typename TFun>
	void for_each(TFun const & fun) const
	{
		if (empty())
		{
			return;
		}
		else if (is_simply())
		{
			for (auto const &s : *this)
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
			for (auto s : *this)
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
	template<typename TFun>
	void for_each(this_type const& other, TFun const & fun) const
	{

		//FIXME NEED OPTIMIZE

		if (is_null())
		{
			return;
		}
		else if (is_simply() && other.is_simply())
		{
			range_type r = *this & other;

			for (auto s : r)
			{
				fun(s);
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

	template<typename TI>
	bool in_box(TI const & idx) const
	{
		return range_type::in_box(idx);
	}

	bool in_box(point_type const & x) const
	{
		return range_type::in_box(mesh_type::coordinates_to_topology(x));
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
			m_mesh_.range(m_mesh_.template sub_index_to_id<iform>()).swap(
					*this);
		}
	}
	void clear_ids()
	{
		if (m_id_set_.size() > 0)
		{
			UNIMPLEMENTED2("TODO: clear ids that out of bound.");
		}
	}

	void reset(point_type const & b, point_type const & e)
	{
		range_type::reset(m_mesh_.coordinates_to_topology(b),
				m_mesh_.coordinates_to_topology(e));
	}

	template<typename T0, typename T1>
	void select(T0 const & b, T1 const & e)
	{
		// FIXME this is incorrect

		index_tuple ib, ie;
		ib = b;
		ie = e;

		range_type(mesh_type::pack_index(ib), mesh_type::pack_index(ie)).swap(
				*this);
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
					return in_box(m_mesh_.coordinates(s));
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

	void remove(std::function<bool(point_type const &)> const & pred)
	{
		filter([&](id_type const & s)
		{
			return !pred(m_mesh_.coordinates(s));
		});
	}

	/**
	 * @name  Data Shape
	 * @{
	 **/

	DataSpace dataspace() const
	{
		DataSpace res = m_mesh_.template dataspace<iform>();

		if (is_simply())
		{
			typename DataSpace::index_tuple b, e, l_b;

			b = mesh_type::unpack_index(std::get<0>(range_type::box()));
			e = mesh_type::unpack_index(std::get<1>(range_type::box()));

			std::tie(l_b, std::ignore) = m_mesh_.local_index_box();

			typename DataSpace::index_tuple offset, count;

			offset = b - l_b;
			offset[ndims] = 0;
			count = e - b;
			count[ndims] = (iform == VERTEX || iform == VOLUME) ? 1 : 3;

			res.select_hyperslab(&offset[0], nullptr, &count[0], nullptr);
		}
		else
		{
			UNIMPLEMENTED;
		}

		return std::move(res);

	}

	/** @}*/

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

}  // namespace simpla

#endif /* CORE_MESH_DOMAIN_H_ */
