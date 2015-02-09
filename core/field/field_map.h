/**
 * @file field_map.h
 *
 *  Created on: 2015年1月30日
 *  @author: salmon
 */

#ifndef CORE_FIELD_FIELD_MAP_H_
#define CORE_FIELD_FIELD_MAP_H_

#include <cstdbool>
#include <memory>
#include <string>

#include "../application/sp_object.h"
#include "../gtl/expression_template.h"

namespace simpla
{
template<typename ...>struct _Field;

namespace _impl
{
struct is_maplike_container;
}  // namespace _impl
/**
 * @ingroup field
 * @brief Field using  associative container 'map'
 */
template<typename TM, typename TContainer>
struct _Field<TM, TContainer, _impl::is_maplike_container> : public SpObject
{
	typedef TM mesh_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef TContainer container_type;
	typedef typename container_type::mapped_type value_type;

	typedef _Field<mesh_type, container_type, _impl::is_maplike_container> this_type;

private:

	mesh_type mesh_;

	std::shared_ptr<container_type> data_;

public:

	template<typename ...Args>
	_Field(mesh_type const & d, Args && ... args) :
			mesh_(d), data_(
					std::make_shared<container_type>(
							std::forward<Args>(args)...))
	{

	}
	_Field(this_type const & that) :
			mesh_(that.mesh_), data_(that.data_)
	{
	}
	~_Field()
	{
	}

	std::string get_type_as_string() const
	{
		return "Field<" + mesh_.get_type_as_string() + ">";
	}

	template<typename TU> using clone_field_type=
	_Field<TM,typename replace_template_type<1,TU,container_type>::type,
	_impl::is_maplike_container>;

	template<typename TU>
	clone_field_type<TU> clone() const
	{
		return clone_field_type<TU>(mesh_);
	}

	mesh_type const & mesh() const
	{
		return mesh_;
	}
	void clear()
	{
		*this = 0;
	}

	template<typename ...Args>
	_Field(this_type & that, Args && ...args) :
			mesh_(that.mesh_, std::forward<Args>(args)...), data_(that.data_)
	{
	}
	bool empty() const
	{
		return mesh_.empty();
	}
	bool is_divisible() const
	{
		return mesh_.is_divisible();
	}

	inline _Field<AssignmentExpression<_impl::_assign, this_type, this_type>> operator =(
			this_type const &that)
	{
		return std::move(
				_Field<
						AssignmentExpression<_impl::_assign, this_type,
								this_type>>(*this, that));
	}

	template<typename TR>
	inline _Field<AssignmentExpression<_impl::_assign, this_type, TR>> operator =(
			TR const &that)
	{
		return std::move(
				_Field<AssignmentExpression<_impl::_assign, this_type, TR>>(
						*this, that));
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		mesh_.pull_back(*data_, fun);
	}

	typedef typename mesh_type::template field_value_type<value_type> field_value_type;

	field_value_type gather(coordinates_type const& x) const
	{
		return std::move(mesh_.gather(*data_, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		mesh_.scatter(*data_, std::forward<Args>(args)...);
	}

//	DataSet dump_data() const
//	{
//		return DataSet();
//	}
public:

	value_type & operator[](id_type const & s)
	{
		return (*data_)[s];
	}
	value_type const & operator[](id_type const & s) const
	{
		return (*data_)[s];
	}

}
;

}  // namespace simpla

#endif /* CORE_FIELD_FIELD_MAP_H_ */
