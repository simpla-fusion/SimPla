/**
 * @file field_sparse.h
 *
 *  Created on: 2015-1-30
 *  @author: salmon
 */

#ifndef COREFieldField_SPARSE_H_
#define COREFieldField_SPARSE_H_

#include <cstdbool>
#include <memory>
#include <string>

#include "../application/sp_object.h"
#include "../toolbox/expression_template.h"

#include "FieldExpression.h"
namespace simpla
{
template<typename ...>struct Field;

/**
 * @ingroup field
 * @brief field using  associative container 'map'
 */
template<typename TM, typename TV>
struct Field<TM, TV, _impl::is_associative_container> : public SpObject
{
	typedef TM mesh_type;

	typedef TV value_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	typedef std::map<id_type, value_type> container_type;

	typedef Field<mesh_type, value_type, _impl::is_associative_container> this_type;

private:

	mesh_type m_mesh_;

	container_type m_dataset_;

public:

	template<typename ...Args>
	Field(mesh_type const & d, Args && ... args)
			: m_mesh_(d)
	{

	}
	Field(this_type const & that)
			: m_mesh_(that.m_mesh_), m_dataset_(that.m_dataset_)
	{
	}

	~Field()
	{
	}

	std::string get_type_as_string() const
	{
		return "field<" + m_mesh_.get_type_as_string() + ">";
	}

	template<typename TU> using cloneField_type= Field<TM,TU, _impl::is_associative_container>;

	template<typename TU>
	cloneField_type<TU> clone() const
	{
		return cloneField_type<TU>(m_mesh_);
	}

	mesh_type const & get_mesh() const
	{
		return m_mesh_;
	}
	void clear()
	{
		m_dataset_->clear();
	}

	bool empty() const
	{
		return m_dataset_.empty();
	}
	bool is_divisible() const
	{
		return m_mesh_.is_divisible();
	}

	inline Field<AssignmentExpression<_impl::_assign, this_type, this_type>> operator =(
			this_type const &that)
	{
		return std::move(
				Field<
						AssignmentExpression<_impl::_assign, this_type,
								this_type>>(*this, that));
	}

	template<typename TR>
	inline Field<AssignmentExpression<_impl::_assign, this_type, TR>> operator =(
			TR const &that)
	{
		return std::move(
				Field<AssignmentExpression<_impl::_assign, this_type, TR>>(
						*this, that));
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		m_mesh_.pull_back(*m_dataset_, fun);
	}

	typedef typename mesh_type::template field_value_type<value_type> field_value_type;

	field_value_type gather(coordinate_tuple const& x) const
	{
		return std::move(m_mesh_.gather(*m_dataset_, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		m_mesh_.scatter(*m_dataset_, std::forward<Args>(args)...);
	}

public:

	value_type & operator[](id_type const & s)
	{
		return (*m_dataset_)[s];
	}
	value_type const & operator[](id_type const & s) const
	{
		return (*m_dataset_)[s];
	}

}
;

}  // namespace simpla

#endif /* COREFieldField_SPARSE_H_ */
