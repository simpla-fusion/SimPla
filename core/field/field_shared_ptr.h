/**
 * @file field_shared_ptr.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef CORE_FIELD_FIELD_SHARED_PTR_H_
#define CORE_FIELD_FIELD_SHARED_PTR_H_

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
template<typename TM, typename TV, typename ...Others>
struct _Field<TM, std::shared_ptr<TV>, Others...> : public SpObject
{

	typedef TM mesh_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef std::shared_ptr<TV> container_type;
	typedef TV value_type;

	typedef _Field<mesh_type, container_type, Others...> this_type;

private:

	mesh_type m_mesh_;

	std::shared_ptr<TV> m_data_;

public:

	_Field(mesh_type const & d) :
			m_mesh_(d), m_data_(nullptr)
	{

	}
	_Field(this_type const & that) :
			m_mesh_(that.m_mesh_), m_data_(that.m_data_)
	{
	}
	~_Field()
	{
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
	_Field<TM,std::shared_ptr<TU>,Others... >;

	template<typename TU>
	clone_field_type<TU> clone() const
	{
		return clone_field_type<TU>(m_mesh_);
	}

	void clear()
	{
		wait_to_ready();

		*this = 0;
	}
	template<typename T>
	void fill(T const &v)
	{
		wait_to_ready();

		std::fill(m_data_.get(), m_data_.get() + m_mesh_.max_hash(), v);
	}

	/** @name range concept
	 * @{
	 */

	template<typename ...Args>
	_Field(this_type & that, Args && ...args) :
			m_mesh_(that.m_mesh_, std::forward<Args>(args)...), m_data_(
					that.m_data_)
	{
	}
	bool empty() const
	{
		return m_data_ == nullptr;
	}
	bool is_divisible() const
	{
		return m_mesh_.is_divisible();
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

//	inline _Field<AssignmentExpression<_impl::_assign, this_type, this_type>> operator =(
//			this_type const &that)
//	{
//		deploy();
//
//		return std::move(
//				_Field<
//						AssignmentExpression<_impl::_assign, this_type,
//								this_type>>(*this, that));
//	}
	inline this_type & operator =(this_type const &that)
	{
		wait_to_ready();

		for (auto s : m_mesh_.range())
		{
			this->operator[](s) = m_mesh_.calculate(that, s);
		}

		return *this;
	}

	template<typename TR>
	inline this_type & operator =(TR const &that)
	{
		wait_to_ready();

//		return std::move(
//				_Field<AssignmentExpression<_impl::_assign, this_type, TR>>(
//						*this, that));

//		parallel_for(mesh_.range(), [&](typename mesh_type::range_type s_range)
//		{
		auto s_range = m_mesh_.range();

		for (auto s : s_range)
		{
			this->operator[](s) = m_mesh_.calculate(that, s);
		}
//		});

		return *this;
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		wait_to_ready();

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

		int ndims = 3;

		nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims;

		std::tie(ndims, l_dims, std::ignore, std::ignore, std::ignore,
				std::ignore) = m_mesh_.dataspace().shape();

		make_send_recv_list(DataType::create<value_type>(), ndims, &l_dims[0],
				m_mesh_.ghost_shape(), &m_send_recv_list_);
	}
private:
	std::vector<mpi_send_recv_s> m_send_recv_list_;
public:

	void sync()
	{
		wait_to_ready();

		VERBOSE << "sync Field" << std::endl;

		sync_update_continue(m_send_recv_list_, m_data_.get(),
				&(SpObject::m_mpi_requests_));
	}

	DataSet dataset() const
	{
		ASSERT(is_ready());

		DataSet res;

		res.data = m_data_;

		res.datatype = DataType::create<value_type>();

		res.dataspace = m_mesh_.dataspace();

		res.properties = properties;

		return std::move(res);
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
namespace _impl
{
class is_sequence_container;
template<typename TContainer> struct field_selector;
template<typename TV>
struct field_selector<std::shared_ptr<TV>>
{
	typedef is_sequence_container type;
};

}  // namespace _impl
/**@} */

template<typename TM, typename TV>
auto make_field(TM const & mesh)
DECL_RET_TYPE(( _Field<TM,std::shared_ptr<TV>>(mesh)))
}
// namespace simpla

#endif /* CORE_FIELD_FIELD_SHARED_PTR_H_ */
