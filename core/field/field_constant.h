/**
 * @file field_constant.h
 *
 *  created on: 2012-3-15
 *      Author: salmon
 */

#ifndef FIELD_CONSTANT_H_
#define FIELD_CONSTANT_H_

namespace simpla
{
template<typename ...> struct _Field;

namespace _impl
{

class this_is_constant;

}  // namespace _impl

template<typename TM, typename TV>
class _Field<TM, TV, _impl::this_is_constant>
{
	typedef _Field<TM, TV, _impl::this_is_constant> this_type;

	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::id_type id_type;
	typedef TV value_type;

private:
	mesh_type m_mesh_;
	value_type m_value_;
public:

	_Field(mesh_type const &m, value_type const &f)
			: m_mesh_(m), m_value_(f)
	{

	}

	_Field(this_type const &other)
			: m_mesh_(other.m_mesh_), m_value_(other.m_value_)
	{
	}

	~_Field()
	{
	}

	void swap(this_type &other)
	{
		std::swap(m_mesh_, other.m_mesh_);
		std::swap(m_value_, other.m_value_);
	}

	this_type &operator=(this_type const &other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	auto operator[](id_type const &s) const DECL_RET_TYPE(m_mesh_.sample(s, m_value_))

	value_type operator()(coordinates_type const &x) const
	{
		return m_value_;
	}

};

template<typename TM, typename TV>
_Field<TM, TV, _impl::this_is_constant> make_field_constant(TM const &m,
		TV const &v)
{
	return std::move(_Field<TM, TV, _impl::this_is_constant>(m, v));
}

}
// namespace simpla

#endif /* FIELD_CONSTANT_H_ */
