/**
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */
#ifndef CORE_FIELD_FIELD_H_
#define CORE_FIELD_FIELD_H_

#include <cstdbool>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "../utilities/utilities.h"
#include "../data_representation/data_interface.h"
#include "../application/sp_object.h"
#include "../gtl/expression_template.h"

#ifdef USE_TBB
#include <tbb/concurrent_unordered_map.h>
#endif

namespace simpla
{

/**
 *@ingroup physical_object
 *@addtogroup field Field
 *@brief @ref field is an abstraction from physical field on 4d or 3d @ref configuration_space
 *
 * ## Summary
 *  - @ref field  assigns a scalar/vector/tensor to each point of a mathematical space (typically a Euclidean space or manifold).
 *  - @ref field  is function, \f$ y=f(x)\f$ , on the  @ref configuration_space, where \f$x\in D\f$ is
 *       coordinates defined in @ref domain \f$D\f$, and  \f$ y \f$ is a scalar/vector/tensor.
 *
 * ## Member types
 *  Member type	 				    | Semantics
 *  --------------------------------|--------------
 *  coordinates_type				| Datatype of coordinates
 *  index_type						| Datatype of of grid points index
 *  value_type 					    | Datatype of value
 *  domain_type					    | Domain
 *  storage_type					| container type
 *
 *
 * ## Member functions
 *
 * ###Constructor
 *
 *  Pseudo-Signature 	 			| Semantics
 *  --------------------------------|--------------
 *  `Field()`						| Default constructor
 *  `~Field() `					    | destructor.
 *  `Field( const Field& ) `	    | copy constructor.
 *  `Field( Field && ) `			| move constructor.
 *
 *
 * ### Domain &  Split
 *
 *  Pseudo-Signature 	 			        | Semantics
 *  ----------------------------------------|--------------
 *  `Field( Domain & D ) `			        | Construct a field on domain \f$D\f$.
 *  `Field( Field &r,split)`			    | Split field into two part,  see @ref concept_domain
 *  `domain_type const &domain() const `	| Get define domain of field
 *  `void domain(domain_type cont&) ` 	    | Reset define domain of field
 *  `Field split(domain_type d)`			| Sub-field on  domain \f$D \cap D_0\f$
 *  `Field boundary()`				        | Sub-field on  boundary \f${\partial D}_0\f$
 *
 * ###   Capacity
 *  Pseudo-Signature 	 			| Semantics
 *  --------------------------------|--------------
 *  `bool empty() `  				| _true_ if memory is not allocated.
 *  `allocate()`					| allocate memory
 *  `data()`						| direct access to the underlying memory
 *  `clear()`						| set value to zero, allocate memory if empty() is _true_
 *
 *
 * ### Element access
 *  Pseudo-Signature 				            | Semantics
 *  --------------------------------------------|--------------
 *  `value_type & at(index_type s)`   			| access element on the grid points _s_ with bounds checking
 *  `value_type & operator[](index_type s) `  	| access element on the grid points _s_as
 *  `field_value_type  operator()(coordiantes_type x) const` | field value on coordinates \f$x\f$, which is interpolated from discrete points
 *
 * ### Assignment
 *   Pseudo-Signature 	 				         | Semantics
 *  ---------------------------------------------|--------------
 *  `Field & operator=(Function const & f)`  	 | assign values as \f$y[s]=f(x)\f$
 *  `Field & operator=(FieldExpression const &)` | Assign operation,
 *  `Field operator=( const Field& )`            | copy-assignment operator.
 *  `Field operator=( Field&& )`		         | move-assignment operator.
 *  `Field & operator+=(Expression const &)`     | Assign operation +
 *  `Field & operator-=(Expression const &)`     | Assign operation -
 *  `Field & operator/=(Expression const &)`     | Assign operation /
 *  `Field & operator*=(Expression const &)`     | Assign operation *
 *
 * ## Non-member functions
 *  Pseudo-Signature  				| Semantics
 *  --------------------------------|--------------
 *  `swap(Field &,Field&)`			| swap
 *
 * ## See also
 *  - @ref FETL
 *  - @ref manifold_concept
 *  - @ref domain_concept
 *
 *
 */
/**
 *  @ingroup field
 *
 *  @brief skeleton of Field data holder
 *   Field is an Associate container
 *     f[id_type i] => value at the discrete point/edge/face/volume i
 *   Field is a Function
 *     f(coordinates_type x) =>  field value(scalar/vector/tensor) at the coordinates x
 *   Field is a Expression
 */
template<typename ... >struct _Field;

template<typename TM, typename TV>
struct _Field<TM, std::vector<TV>> : public SpObject,
		public enable_create_from_this<_Field<TM, std::vector<TV> > >
{

	typedef TM mesh_type;

	typedef TV value_type;

	typedef typename mesh_type::id_type id_type;

	typedef _Field<mesh_type, container_type> this_type;

private:
	mesh_type mesh_;
public:

	_Field(mesh_type const & d) :
			mesh_(d)
	{
	}
	_Field(this_type const & that) :
			container_type(that), mesh_(that.mesh_)
	{
	}
	~_Field()
	{
	}

	std::string get_type_as_string() const
	{
		return "Field<" + mesh_type::get_type_as_string() + ">";
	}

	this_type & self()
	{
		return *this;
	}
	this_type const & self() const
	{
		return *this;
	}
	value_type & operator[](id_type const & s)
	{
		return data_[mesh_.hash(s)];
	}
	value_type const & operator[](id_type const & s) const
	{
		return data_[mesh_.hash(s)];
	}

	/**
	 * @name assignment
	 * @{
	 */

	inline this_type &
	operator =(this_type const &that)
	{

		mesh_.foreach(_impl::_assign(), *this, that);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const&that)
	{

		mesh_.foreach(_impl::_assign(), *this, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{

		mesh_.foreach(_impl::plus_assign(), *this, that);
		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{

		mesh_.foreach(_impl::minus_assign(), *this, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{

		mesh_.foreach(_impl::multiplies_assign(), *this, that);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{

		mesh_.foreach(_impl::divides_assign(), *this, that);
		return (*this);
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{

		mesh_.pull_back(*this, fun);
	}

	/** @} */

	typedef typename field_traits<this_type>::field_value_type field_value_type;

	field_value_type gather(typename mesh_type::coordinates_type const& x) const
	{
		return std::move(mesh_.gather(*this, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		mesh_.scatter(*this, std::forward<Args>(args)...);
	}

}
;
template<typename TM, typename TContainer>
struct _Field<TM, TContainer> : public SpObject,
		public TContainer,
		enable_create_from_this<_Field<TM, TContainer>>
{

	typedef TM mesh_type;

	typedef TContainer container_type;

	typedef typename container_traits<container_type>::value_type value_type;

	typedef typename mesh_type::id_type id_type;

	typedef _Field<mesh_type, container_type> this_type;

private:
	mesh_type mesh_;
public:

	_Field(mesh_type const & d) :
			mesh_(d)
	{
	}
	_Field(this_type const & that) :
			container_type(that), mesh_(that.mesh_)
	{
	}
	~_Field()
	{
	}

	std::string get_type_as_string() const
	{
		return "Field<" + mesh_type::get_type_as_string() + ">";
	}

	this_type & self()
	{
		return *this;
	}
	this_type const & self() const
	{
		return *this;
	}
	/**
	 * @name assignment
	 * @{
	 */

	inline this_type &
	operator =(this_type const &that)
	{

		mesh_.foreach(_impl::_assign(), *this, that);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const&that)
	{

		mesh_.foreach(_impl::_assign(), *this, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{

		mesh_.foreach(_impl::plus_assign(), *this, that);
		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{

		mesh_.foreach(_impl::minus_assign(), *this, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{

		mesh_.foreach(_impl::multiplies_assign(), *this, that);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{

		mesh_.foreach(_impl::divides_assign(), *this, that);
		return (*this);
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{

		mesh_.pull_back(*this, fun);
	}

	/** @} */

	typedef typename field_traits<this_type>::field_value_type field_value_type;

	field_value_type gather(typename mesh_type::coordinates_type const& x) const
	{
		return std::move(mesh_.gather(*this, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		mesh_.scatter(*this, std::forward<Args>(args)...);
	}

}
;

template<typename TM, typename TField>
struct _Field<TField &> : public SpObject,
		public enable_create_from_this<TField>
{

	typedef TField field_type;

	typedef typename field_type::mesh_type mesh_type;

	typedef typename field_type::value_type value_type;

	typedef typename mesh_type::id_type id_type;

	typedef _Field<field_type &> this_type;

private:
	mesh_type mesh_;
	std::shared_ptr<field_type> root_;
public:

	template<typename ...Args>
	_Field(field_type & that, Args && ...args) :
			mesh_(that.mesh_, std::forward<Args>(args)...), root_(
					that.root_holder())
	{
	}
	~_Field()
	{
	}

	std::string get_type_as_string() const
	{
		return "Field<" + mesh_type::get_type_as_string() + ">";
	}

	/**
	 * @name splittable container
	 * @{
	 */
	_Field(this_type & that, op_split) :
			root_(that.root_holder()), mesh_(that.mesh_, op_split())
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

	/**
	 * @}
	 */

	/**
	 * @name assignment
	 * @{
	 */

	inline this_type &
	operator =(this_type const &that)
	{
		mesh_.foreach(_impl::_assign(), *root_, that);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const&that)
	{

		mesh_.foreach(_impl::_assign(), *root_, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{

		mesh_.foreach(_impl::plus_assign(), *root_, that);
		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{

		mesh_.foreach(_impl::minus_assign(), *root_, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{

		mesh_.foreach(_impl::multiplies_assign(), *root_, that);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{

		mesh_.foreach(_impl::divides_assign(), *root_, that);
		return (*this);
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{

		mesh_.pull_back(*root_, fun);
	}

	/** @} */

	typedef typename field_traits<this_type>::field_value_type field_value_type;

	field_value_type gather(typename mesh_type::coordinates_type const& x) const
	{
		return std::move(mesh_.gather(*root_, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		mesh_.scatter(*root_, std::forward<Args>(args)...);
	}

}
;

template<typename TM, typename TC, typename ...Others>
struct reference_traits<_Field<TM, TC, Others...> >
{
	typedef _Field<TM, TC, Others...> const & type;
};

template<typename TV, typename TM> using Field= _Field< TM,std::shared_ptr<TV> >;

template<typename > struct is_field
{
	static constexpr bool value = false;
};

template<typename ...T> struct is_field<_Field<T...>>
{
	static constexpr bool value = true;
};

template<typename ...> struct field_traits;

template<typename T> struct field_traits<T>
{

	static constexpr size_t ndims = 0;

	static constexpr size_t iform = VERTEX;

	typedef T value_type;

	typedef T field_value_type;

	static constexpr bool is_field = false;

};

template<typename ...T>
struct field_traits<_Field<T ...>>
{
	static constexpr bool is_field = true;

	typedef typename _Field<T ...>::mesh_type mesh_type;

	static constexpr size_t ndims = mesh_type::ndims;

	static constexpr size_t iform = mesh_type::iform;

	typedef typename _Field<T ...>::value_type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;

};

/// @name  Field Expression
/// @{

template<typename TOP, typename TL, typename TR> struct Expression;

template<typename TOP, typename TL>
struct field_traits<_Field<Expression<TOP, TL, std::nullptr_t> >>
{
private:
	typedef typename field_traits<TL>::value_type l_type;
public:
	static constexpr bool is_field = true;

	static constexpr size_t ndims = field_traits<TL>::ndims;

	static constexpr size_t iform = field_traits<TL>::iform;

	typedef typename sp_result_of<TOP(l_type)>::type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;

};

template<typename TOP, typename TL, typename TR>
struct field_traits<_Field<Expression<TOP, TL, TR> > >
{
private:
	static constexpr size_t NDIMS = sp_max<size_t, field_traits<TL>::ndims,
			field_traits<TL>::ndims>::value;

	static constexpr size_t IL = field_traits<TL>::iform;
	static constexpr size_t IR = field_traits<TR>::iform;

	typedef typename field_traits<TL>::value_type l_type;
	typedef typename field_traits<TR>::value_type r_type;

public:
	static const size_t ndims = NDIMS;
	static const size_t iform = IL;
	static constexpr bool is_field = true;

	typedef typename sp_result_of<TOP(l_type, r_type)>::type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;

};

template<typename TOP, typename TL, typename TR>
struct _Field<Expression<TOP, TL, TR>> : public Expression<TOP, TL, TR>
{
	typedef _Field<Expression<TOP, TL, TR>> this_type;

	using Expression<TOP, TL, TR>::Expression;
};

template<typename TOP, typename TL, typename TR>
struct _Field<BooleanExpression<TOP, TL, TR>> : public Expression<TOP, TL, TR>
{
	typedef _Field<BooleanExpression<TOP, TL, TR>> this_type;

	using Expression<TOP, TL, TR>::Expression;

	operator bool() const
	{
		UNIMPLEMENTED;
		return false;
	}
};

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)

#define SP_DEF_BINOP_FIELD_NTUPLE(_OP_,_NAME_)                                                 \
template<typename ...T1, typename T2, size_t ... N>                                            \
_Field<Expression<_impl::plus, _Field<T1...>, nTuple<T2, N...> > > operator _OP_(              \
		_Field<T1...> const & l, nTuple<T2, N...> const &r)                                    \
{return (_Field<Expression<_impl::_NAME_, _Field<T1...>, nTuple<T2, N...> > >(l, r));}         \
template<typename T1, size_t ... N, typename ...T2>                                            \
_Field<Expression<_impl::plus, nTuple<T1, N...>, _Field<T2...> > > operator _OP_(              \
		nTuple<T1, N...> const & l, _Field< T2...>const &r)                                    \
{	return (_Field<Expression< _impl::_NAME_,T1,_Field< T2...>>>(l,r));}                       \


SP_DEF_BINOP_FIELD_NTUPLE(+, plus)
SP_DEF_BINOP_FIELD_NTUPLE(-, minus)
SP_DEF_BINOP_FIELD_NTUPLE(*, multiplies)
SP_DEF_BINOP_FIELD_NTUPLE(/, divides)
SP_DEF_BINOP_FIELD_NTUPLE(%, modulus)
SP_DEF_BINOP_FIELD_NTUPLE(^, bitwise_xor)
SP_DEF_BINOP_FIELD_NTUPLE(&, bitwise_and)
SP_DEF_BINOP_FIELD_NTUPLE(|, bitwise_or)
#undef SP_DEF_BINOP_FIELD_NTUPLE

template<typename TV, typename TD>
auto make_field(TD const& d)
DECL_RET_TYPE((_Field<TV,TD >( (d) )))

template<typename, size_t> class Domain;

template<typename TV, size_t IFORM, typename TM>
_Field<TV, Domain<TM, IFORM> > make_form(std::shared_ptr<TM> manifold)
{
	return std::move(_Field<TV, Domain<TM, IFORM> >(manifold));
}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_H_ */
