/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef CORE_FIELD_FIELD_DENSE_H_
#define CORE_FIELD_FIELD_DENSE_H_

#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../application/sp_object.h"
#include "../dataset/dataset.h"
#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/properties.h"
#include "../gtl/type_traits.h"
#include "../parallel/mpi_update.h"

#include "field_expression.h"

namespace simpla
{

template<typename ...> struct _Field;

/**
 * @ingroup field
 * @{
 */

/**
 *  Simple Field
 */
template<typename TM, typename TV>
struct _Field<TM, TV, _impl::is_sequence_container> : public SpObject
{

	typedef TM mesh_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef TV value_type;

	typedef std::shared_ptr<value_type> container_type;

	typedef _Field<mesh_type, value_type, _impl::is_sequence_container> this_type;

private:

	mesh_type m_mesh_;

	std::shared_ptr<TV> m_data_;

public:

	_Field(mesh_type const & d) :
			m_mesh_(d), m_data_(nullptr)
	{
	}
	_Field(this_type const & other) :
			m_mesh_(other.m_mesh_), m_data_(other.m_data_)
	{
	}
	_Field(this_type && other) :
			m_mesh_(other.m_mesh_), m_data_(other.m_data_)
	{
	}
	~_Field()
	{
	}
	void swap(this_type & other)
	{
		std::swap(m_mesh_, other.m_mesh_);
		std::swap(m_data_, other.m_data_);
	}
	std::string get_type_as_string() const
	{
		return "Field<" + m_mesh_.get_type_as_string() + ">";
	}
	mesh_type const & mesh() const
	{
		return m_mesh_;
	}

	std::shared_ptr<value_type> data()
	{
		return m_data_;
	}

	template<typename TU> using clone_field_type=
	_Field<TM,TU, _impl::is_sequence_container >;

	template<typename TU>
	clone_field_type<TU> clone() const
	{
		return clone_field_type<TU>(m_mesh_);
	}

	template<typename TRange>
	this_type sub_field(TRange const & r) const
	{
		this_type other(*this);

		other.m_mesh_.ids(r);

		return std::move(other);
	}

	void clear()
	{
		wait();

		*this = 0;
	}
	template<typename T>
	void fill(T const &v)
	{
		wait();

		std::fill(m_data_.get(), m_data_.get() + m_mesh_.max_hash(), v);
	}

	/** @name range concept
	 * @{
	 */

	bool empty() const
	{
		return m_data_ == nullptr;
	}

	bool is_valid() const
	{
		return m_data_ != nullptr;
	}
	/**@}*/

	/**
	 * @name assignment
	 * @{
	 */

	inline this_type & operator =(this_type const &that)
	{
		wait();

		auto s_range = m_mesh_.range();

		for (auto s : s_range)
		{
			this->operator[](s) = m_mesh_.calculate(that, s);
		}

		return *this;
	}

	template<typename TR>
	inline this_type & operator =(TR const &that)
	{
		wait();

		auto s_range = m_mesh_.range();

		for (auto s : s_range)
		{
			this->operator[](s) = m_mesh_.calculate(that, s);
		}

		return *this;
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		wait();

		m_mesh_.pull_back(*this, fun);
	}

	/** @} */

	/** @name access
	 *  @{*/

	typedef typename mesh_type::template field_value_type<value_type> field_value_type;

	field_value_type gather(coordinates_type const& x) const
	{
		return std::move(m_mesh_.gather(*this, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		m_mesh_.scatter(*this, std::forward<Args>(args)...);
	}

	/**@}*/

	void deploy()
	{
		if (m_data_ == nullptr)
		{
			m_data_ = sp_make_shared_array<value_type>(m_mesh_.max_hash());
		}

		SpObject::prepare_sync(m_mesh_.ghost_shape());
	}

	DataSet dataset() const
	{
		ASSERT(is_ready());

		DataSet res;

		res.data = m_data_;

		res.datatype = DataType::create<value_type>();

		res.dataspace = m_mesh_.dataspace();

		res.properties = SpObject::properties;

		return std::move(res);
	}

	template<typename TFun>
	void for_each(TFun const& fun)
	{
		wait();

		auto s_range = m_mesh_.range();

		for (auto s : s_range)
		{
			fun(m_data_.get()[m_mesh_.hash(s)]);
		}

	}
	template<typename TFun>
	void for_each(TFun const& fun) const
	{
		ASSERT(is_ready());

		auto s_range = m_mesh_.range();

		for (auto s : s_range)
		{
			fun(m_data_.get()[m_mesh_.hash(s)]);
		}
	}

public:
	template<typename IndexType>
	value_type & operator[](IndexType const & s)
	{
		return m_data_.get()[m_mesh_.hash(s)];
	}
	template<typename IndexType>
	value_type const & operator[](IndexType const & s) const
	{
		return m_data_.get()[m_mesh_.hash(s)];
	}

	template<typename ...Args>
	auto id(Args && ... args)
	DECL_RET_TYPE((m_data_.get()[m_mesh_.hash(std::forward<Args>(args)...)]))

	template<typename ...Args>
	auto id(Args && ... args) const
	DECL_RET_TYPE((m_data_.get()[m_mesh_.hash(std::forward<Args>(args)...)]))

	template<typename ...Args>
	auto operator()(Args && ... args) const
	DECL_RET_TYPE((m_mesh_.gather(*this,std::forward<Args>(args)...)))

}
;

/**@} */

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_DENSE_H_ */
