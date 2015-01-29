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
#include "../gtl/containers/container_traits.h"

namespace simpla
{

/**
 *@ingroup physical_object
 *@addtogroup field Field
 *@brief @ref field is an abstraction from physical field on 4d or 3d @ref configuration_space
 *
 * ## Summary
 *  - @ref field  assigns a scalar/vector/tensor to each point of a
 *     mathematical space (typically a Euclidean space or manifold).
 *
 *  - @ref field  is function, \f$ y=f(x)\f$ , on the
 *     @ref configuration_space, where \f$x\in D\f$ is
 *       coordinates defined in @ref domain \f$D\f$, and  \f$ y \f$
 *       is a scalar/vector/tensor.
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
 *  - @ref mesh_concept
 *  - @ref diff_geometry
 *
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

template<typename TM, typename TContainer>
struct _Field<TM, TContainer> : public SpObject
{

	typedef TM mesh_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef TContainer container_type;
	typedef typename container_traits<container_type>::value_type value_type;

	typedef _Field<mesh_type, container_type> this_type;

private:

	static constexpr bool is_associative_container = false;

//	container_traits<container_type>::is_associative_container;

	mesh_type mesh_;

	typename container_traits<container_type>::holder_type data_;

public:

	_Field(mesh_type const & d) :
			mesh_(d), data_(nullptr)
	{
//		static_assert(is_indexable<container_type,id_type>::value ||
//				is_indexable<container_type,size_t>::value,"illegal container type" );

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
	mesh_type const & mesh() const
	{
		return mesh_;
	}
	void clear()
	{
		allocate();
		container_traits<container_type>::clear(data_, mesh_.max_hash());
	}

	/** @name range concept
	 * @{
	 */

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

	/**@}*/

	/**
	 * @name assignment
	 * @{
	 */

	inline _Field<AssignmentExpression<_impl::_assign, this_type, this_type>> operator =(
			this_type const &that)
	{
		allocate();
		return std::move(
				_Field<
						AssignmentExpression<_impl::_assign, this_type,
								this_type>>(*this, that));
	}

	template<typename TR>
	inline _Field<AssignmentExpression<_impl::_assign, this_type, TR>> operator =(
			TR const &that)
	{
		allocate();
		return std::move(
				_Field<AssignmentExpression<_impl::_assign, this_type, TR>>(
						*this, that));
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		allocate();
		mesh_.pull_back(*data_, fun);
	}

	/** @} */

	/** @name access
	 *  @{*/

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

	value_type & operator[](id_type const & s)
	{
		return get(s);
	}
	value_type const & operator[](id_type const & s) const
	{
		return get(s);
	}

	template<typename ...Args>
	auto operator()(Args && ...s)
	DECL_RET_TYPE((get( std::forward<Args>(s)...)))

	template<typename ...Args>
	auto operator()(Args && ...s) const
	DECL_RET_TYPE((get( std::forward<Args>(s)...)))

	/**@}*/

//	DataSet dump_data() const
//	{
//		return DataSet();
//	}
private:
	void allocate()
	{
		if (data_ == nullptr)
		{
			data_ = container_traits<container_type>::allocate(
					mesh_.max_hash());
		}
	}
	template<typename ...Args>
	value_type & get(Args && ...s)
	{
		return get_value(data_, mesh_.hash(std::forward<Args>(s)...));
	}
	template<typename ...Args>
	value_type const& get(Args && ...s) const
	{
		return get_value(data_, mesh_.hash(std::forward<Args>(s)...));
	}
//	auto get(id_type const & s)->
//	typename std::enable_if< is_associative_container,value_type & >::type
//	{
//		return data_->operator[](s);
//	}
//	auto get(id_type const & s) const->
//	typename std::enable_if< is_associative_container,value_type const & >::type
//	{
//		return data_->operator[](s);
//	}
//	template<typename ...Args>
//	auto get(Args && ...s)->
//	typename std::enable_if< is_associative_container,value_type & >::type
//	{
//		return data_->operator[](mesh_.id(std::forward<Args>(s)...));
//	}
//	template<typename ...Args>
//	auto get(Args && ...s) const->
//	typename std::enable_if< is_associative_container,value_type const & >::type
//	{
//		return data_->operator[](mesh_.id(std::forward<Args>(s)...));
//	}

}
;

template<typename TM, typename TC, typename ...Others>
struct reference_traits<_Field<TM, TC, Others...> >
{
	typedef _Field<TM, TC, Others...> const & type;
};

template<typename TM, typename TV> using Field= _Field< TM,std::shared_ptr<TV> >;

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

	typedef std::nullptr_t mesh_type;

	typedef T value_type;

	static constexpr bool is_field = false;

};

template<typename ...T>
struct field_traits<_Field<T ...>>
{
	static constexpr bool is_field = true;

	typedef typename _Field<T ...>::mesh_type mesh_type;

	typedef typename _Field<T ...>::value_type value_type;

};

/// @name  Field Expression
/// @{

template<typename ...>class Expression;
template<typename ...>class BooleanExpression;

template<typename TOP, typename TL>
struct _Field<Expression<TOP, TL, std::nullptr_t>> : public Expression<TOP, TL,
		std::nullptr_t>
{
	typedef typename field_traits<TL>::value_type l_type;
public:

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef typename sp_result_of<TOP(l_type)>::type value_type;

	typedef _Field<Expression<TOP, TL, std::nullptr_t>> this_type;

	using Expression<TOP, TL, std::nullptr_t>::Expression;
};

template<typename TOP, typename TL, typename TR>
struct _Field<Expression<TOP, TL, TR>> : public Expression<TOP, TL, TR>
{

	typedef typename field_traits<TL>::value_type l_type;
	typedef typename field_traits<TR>::value_type r_type;

public:

	typedef typename sp_result_of<TOP(l_type, r_type)>::type value_type;

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef _Field<Expression<TOP, TL, TR>> this_type;

	using Expression<TOP, TL, TR>::Expression;
};

template<typename TOP, typename TL, typename TR>
struct _Field<BooleanExpression<TOP, TL, TR>> : public Expression<TOP, TL, TR>
{
	typedef bool value_type;

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef _Field<BooleanExpression<TOP, TL, TR>> this_type;

	using Expression<TOP, TL, TR>::Expression;

	operator bool() const
	{
		UNIMPLEMENTED;
		return false;
	}
};

template<typename TOP, typename TL, typename TR>
struct _Field<AssignmentExpression<TOP, TL, TR>> : public AssignmentExpression<
		TOP, TL, TR>
{
	typedef AssignmentExpression<TOP, TL, TR> expression_type;

	typedef typename field_traits<TL>::value_type value_type;

	typedef typename field_traits<TL>::mesh_type mesh_type;

	typedef _Field<AssignmentExpression<TOP, TL, TR>> this_type;

	using AssignmentExpression<TOP, TL, TR>::AssignmentExpression;

	bool is_excuted_ = false;

	void excute()
	{
		if (!is_excuted_)
		{
			expression_type::lhs.mesh().calculate(*this);
			is_excuted_ = true;
		}

	}
	void do_not_excute()
	{
		is_excuted_ = true;
	}

	~_Field()
	{
		excute();
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

template<typename TV, typename TM>
auto make_field(TM const& mesh)
DECL_RET_TYPE((_Field<TM,std::shared_ptr<TV> >( mesh )))

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_H_ */
