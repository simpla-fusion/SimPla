/*
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
#include "../parallel/parallel.h"
#include "../utilities/utilities.h"
#include "../physics/physical_object.h"

#include "../data_interface/data_set.h"
#include "../design_pattern/expression_template.h"

namespace simpla
{

/**
 *@ingroup physical_object
 *@addtogroup field Field
 *
 *## Summary
 * \note A _Field_ assigns a scalar/vector/tensor to each point of a mathematical space (typically a Euclidean space or manifold).
 *
 * \note A _Field_ is a map / function \f$y=f(x)\f$, where \f$x\in D\f$ is coordinates defined in _domain_ \f$D\f$, and \f$y\f$ is a scalar/vector/tensor.
 *
 *## Member types
 * Member type	 				| Semantics
 * -------------------------------|--------------
 * coordinates_type				| Datatype of coordinates
 * index_type						| Datatype of of grid points index
 * value_type 					| Datatype of value
 * domain_type					| Domain
 * storage_type					| container type
 *
 *
 *## Member functions
 *
 *###Constructor
 *
 * Pseudo-Signature 	 			| Semantics
 * -------------------------------|--------------
 * `Field()`						| Default constructor
 * `~Field() `					| destructor.
 * `Field( const Field& ) `	| copy constructor.
 * `Field( Field && ) `			| move constructor.
 *
 *
 *### Domain &  Split
 *
 * Pseudo-Signature 	 			| Semantics
 * -------------------------------|--------------
 * `Field( Domain & D ) `			| Construct a field on domain \f$D\f$.
 * `Field( Field &r,split)`			| Split field into two part,  see @ref concept_domain
 * `domain_type const &domain() const `	| Get define domain of field
 * `void domain(domain_type cont&) ` 	| Reset define domain of field
 * `Field split(domain_type d)`			| Sub-field on  domain \f$D \cap D_0\f$
 * `Field boundary()`				| Sub-field on  boundary \f${\partial D}_0\f$
 *
 *###   Capacity
 * Pseudo-Signature 	 				| Semantics
 * -------------------------------|--------------
 * `bool empty() `  				| _true_ if memory is not allocated.
 * `allocate()`					| allocate memory
 * `data()`						| direct access to the underlying memory
 * `clear()`						| set value to zero, allocate memory if empty() is _true_
 *
 *
 *### Element access
 * Pseudo-Signature 				| Semantics
 * -------------------------------|--------------
 * `value_type & at(index_type s)`   			| access element on the grid points _s_ with bounds checking
 * `value_type & operator[](index_type s) `  	| access element on the grid points _s_as
 * `field_value_type  operator()(coordiantes_type x) const` | field value on coordinates \f$x\f$, which is interpolated from discrete points
 *
 *### Assignment
 *  Pseudo-Signature 	 				| Semantics
 * -------------------------------|--------------
 * `Field & operator=(Function const & f)`  	| assign values as \f$y[s]=f(x)\f$
 * `Field & operator=(FieldExpression const &)` | Assign operation,
 * `Field operator=( const Field& )`| copy-assignment operator.
 * `Field operator=( Field&& )`		| move-assignment operator.
 * `Field & operator+=(Expression const &)` | Assign operation +
 * `Field & operator-=(Expression const &)` | Assign operation -
 * `Field & operator/=(Expression const &)` | Assign operation /
 * `Field & operator*=(Expression const &)` | Assign operation *
 *
 *## Non-member functions
 * Pseudo-Signature  				| Semantics
 * -------------------------------|--------------
 * `swap(Field &,Field&)`			| swap
 *
 *## See also
 * - @ref FETL
 * - @ref manifold_concept
 * - @ref domain_concept
 *
 *
 */

template<typename ... >struct _Field;

/**
 *  @ingroup physical_object
 *
 *  @brief skeleton of Field data holder
 *   Field is an Associate container
 *     f[id_type i] => value at the discrete point/edge/face/volume i
 *   Field is a Function
 *     f(coordinates_type x) =>  field value(scalar/vector/tensor) at the coordinates x
 *   Field is a Expression
 */
template<typename TV, typename TDomain>
struct _Field<TV, TDomain> : public PhysicalObject, public TDomain
{

	typedef TDomain domain_type;
	typedef typename domain_type::id_type id_type;

	typedef TV value_type;
	typedef typename domain_type::template container_type<value_type> container_type;
	typedef _Field<value_type, domain_type> this_type;

private:

	container_type data_;

public:

	template<typename ...Args>
	_Field(Args &&...args) :
			domain_type(std::forward<Args>(args)...), data_(nullptr)
	{
	}
	_Field(this_type const & that) :
			domain_type(that), data_(that.data_)
	{
	}

	~_Field()
	{
	}

	std::string get_type_as_string() const
	{
//		" + domain_type::get_type_as_string() + "
		return "Field<>";
	}

	void swap(this_type &r)
	{
		sp_swap(r.data_, data_);
		domain_type::swap(r);
	}

	container_type & data()
	{
		return data_;
	}

	container_type const & data() const
	{
		return data_;
	}

	void data(container_type d)
	{
		sp_swap(d, data_);
	}

	bool is_same(this_type const & r) const
	{
		return data_ == r.data_;
	}
	template<typename TR>
	bool is_same(TR const & r) const
	{
		return data_ == r.data_;
	}

	bool empty() const
	{
		return data_ == nullptr;
	}
	bool is_valid() const
	{
		return !empty();
	}
	void allocate()
	{
		if (data_ == nullptr)
		{
			auto lock = parallel::lock_guard(mutex_);

			domain_type::template allocate<value_type>().swap(data_);

			PhysicalObject::update();
		}
	}

	void clear()
	{
		allocate();
		container_traits<container_type>::clear(data_, domain_type::size());
	}

	domain_type & domain()
	{
		return *this;
	}
	domain_type const & domain() const
	{
		return *this;
	}
	DataSet dataset() const
	{
		return std::move(domain_type::dataset(data_, properties()));
	}

	/***
	 * @name Access operation
	 * @{
	 */
	value_type & get(id_type const &s)
	{
		return domain_type::access(data_, s);
	}

	value_type const& get(id_type const &s) const
	{
		return domain_type::access(data_, s);
	}

	value_type & operator[](id_type const & s)
	{
		return domain_type::access(data_, s);
	}

	value_type const & operator[](id_type const & s) const
	{
		return domain_type::access(data_, s);
	}

	/** @} */

	/**
	 * @name Assignment
	 * @{
	 */
	inline this_type &
	operator =(this_type const &that)
	{
		allocate();

		domain_type::foreach(_impl::_assign(), data_, that);

		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const&that)
	{
		allocate();

		domain_type::foreach(_impl::_assign(), data_, that);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{
		allocate();

		domain_type::foreach(_impl::plus_assign(), data_, that);

		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{
		allocate();

		domain_type::foreach(_impl::minus_assign(), data_, that);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{
		allocate();
		domain_type::foreach(_impl::multiplies_assign(), data_, that);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{
		allocate();
		domain_type::foreach(_impl::divides_assign(), data_, that);

		return (*this);
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		allocate();
		domain_type::pull_back(data_, fun);
	}

	/** @} */

	typedef typename std::conditional<
			domain_type::iform == VERTEX || domain_type::iform == VOLUME,
			value_type, nTuple<value_type, 3>>::type field_value_type;

	field_value_type gather(
			typename domain_type::coordinates_type const& x) const
	{
		return std::move(domain_type::gather(data_, x));

	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		domain_type::scatter(data_, std::forward<Args>(args)...);
	}

}
;

template<typename TV, typename TC>
struct reference_traits<_Field<TV, TC> >
{
	typedef _Field<TV, TC> const & type;
};

template<typename TV, typename TDomain> using Field= _Field< TV,TDomain >;

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

template<typename TV, typename TD>
struct field_traits<_Field<TV, TD>>
{
	static constexpr bool is_field = true;

	static constexpr size_t ndims = TD::ndims;

	static constexpr size_t iform = TD::iform;

	typedef typename _Field<TV, TD>::value_type value_type;

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
