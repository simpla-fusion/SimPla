//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_PRIMARYOPS_H
#define SIMPLA_PRIMARYOPS_H
namespace simpla { namespace algebra
{
template<typename ...> class Expression;

template<typename ...> struct BooleanExpression;
template<typename ...> struct AssignmentExpression;


//namespace tags
//{
//struct binary_right
//{
//    template<typename TL, typename TR> TR const &operator()(TL const &, TR const &r) const { return r; }
//};
//
//struct null_op
//{
//    template<typename TL> TL const &operator()(TL const &l) const { return l; }
//};
//
//struct _swap
//{
//    template<typename TL, typename TR> void operator()(TL &l, TR &r) const { std::swap(l, r); }
//
//    template<typename TL, typename TR, typename TI> void operator()(TL &l, TR &r, TI const &s) const
//    {
//        std::swap(traits::get_value(l, s), traits::get_value(r, s));
//    }
//};
//
//#define DEF_BOP(_NAME_, _OP_)                                                               \
//struct _NAME_                                                                             \
//{                                                                                              \
//    template<typename TL, typename TR>                                                         \
//    constexpr auto operator()(TL const& l, TR const & r) const->decltype(l _OP_ r)                 \
//    {  return l _OP_ r;   }                                                                       \
//    template<typename TL, typename TR,typename TI>                                                         \
//    constexpr auto operator()(TL const& l, TR const & r,TI const & s) const         \
//    ->decltype(traits::get_value(l,s) _OP_ traits::get_value( r,s) )                 \
//    {  return traits::get_value(l,s) _OP_ traits::get_value( r,s);   }                                    \
//};
//
//#define DEF_UOP(_NAME_, _OP_)                                                                                \
//struct _NAME_                                                                             \
//{                                                                                              \
//    template<typename TL >                                                         \
//    constexpr auto operator()(TL const & l ) const->decltype(_OP_ l )                 \
//    {  return  _OP_ l;   }                                                   \
//    template<typename TL,typename TI >                                                         \
//    constexpr auto operator()(TL const & l, TI const & s) const->decltype(_OP_ traits::get_value( l ,s)) \
//    {  return  _OP_  traits::get_value( l ,s);   } \
//    template<typename TL >                                                         \
//    static constexpr auto eval(TL const & l ) ->decltype(_OP_ l )                 \
//    {  return  _OP_ l;   }                                                   \
//    template<typename TL,typename TI >                                                         \
//    static constexpr auto eval(TL const & l, TI const & s)->decltype(_OP_ traits::get_value( l ,s)) \
//    {  return  _OP_  traits::get_value( l ,s);   } \
//};
//
//DEF_BOP(plus, +)
//
//DEF_BOP(minus, -)
//
//DEF_BOP(multiplies, *)
//
//DEF_BOP(divides, /)
//
//DEF_BOP(modulus, %)
//
//DEF_UOP(negate, -)
//
//DEF_UOP(unary_plus, +)
//
//DEF_BOP(bitwise_and, &)
//
//DEF_BOP(bitwise_or, |)
//
//DEF_BOP(bitwise_xor, ^)
//
//DEF_UOP(bitwise_not, ~)
//
//DEF_BOP(shift_left, <<)
//
//DEF_BOP(shift_right, >>)
//
//DEF_UOP(logical_not, !)
//
//DEF_BOP(logical_and, &&)
//
//DEF_BOP(logical_or, ||)
//
//DEF_BOP(not_equal_to, !=)
//
//DEF_BOP(greater, >)
//
//DEF_BOP(less, <)
//
//DEF_BOP(greater_equal, >=)
//
//DEF_BOP(less_equal, <=)
//
//#undef DEF_UOP
//#undef DEF_BOP
///**
// * ### Assignment Operator
// *
// *   Pseudo-Signature 	 				         | Semantics
// *  ---------------------------------------------|--------------
// *  `operator+=(GeoObject &,Expression const &)`     | Assign operation +
// *  `operator-=(GeoObject & ,Expression const &)`     | Assign operation -
// *  `operator/=(GeoObject & ,Expression const &)`     | Assign operation /
// *  `operator*=(GeoObject & ,Expression const &)`     | Assign operation *
// */
//#define DEF_ASSIGN_OP(_NAME_, _OP_)                                                               \
//struct _NAME_                                                                             \
//{                                                                                              \
//    template<typename TL, typename TR>                                                         \
//      void operator()(TL  & l, TR const & r) const                  \
//    { l _OP_ r ; }           \
//    template<typename TL, typename TR,typename TI>                                                         \
//      void operator()(TL  & l, TR const & r,TI const & s)const           \
//    {    traits::get_value(l,s) _OP_ traits::get_value( r,s)   ;    }    \
//};
//
////DEF_ASSIGN_OP(_assign, =)
//DEF_ASSIGN_OP(plus_assign, +=)
//
//DEF_ASSIGN_OP(minus_assign, -=)
//
//DEF_ASSIGN_OP(multiplies_assign, *=)
//
//DEF_ASSIGN_OP(divides_assign, /=)
//
//DEF_ASSIGN_OP(modulus_assign, %=)
//
//#undef DEF_ASSIGN_OP
//
//struct _assign
//{
//    template<typename TL, typename TR>
//    void operator()(TL &l, TR const &r) const { l = r; }
//
//    template<typename TL, typename TR, typename TI>
//    void operator()(TL &l, TR const &r, TI const &s) const { traits::get_value(l, s) = traits::get_value(r, s); }
//};
//
//struct equal_to
//{
//    template<typename TL, typename TR>
//    constexpr bool operator()(TL const &l, TR const &r) const { return l == r; }
//
//    constexpr bool operator()(double l, double r) const
//    {
//        return std::abs(l - r) <= std::numeric_limits<double>::epsilon();
//    }
//};
//
//template<typename TOP> struct op_traits { typedef logical_and type; };
//
//template<> struct op_traits<not_equal_to> { typedef logical_or type; };
//
//#define DEF_STD_BINARY_FUNCTION(_NAME_)                                                               \
//struct _##_NAME_                                                                             \
//{                                                                                              \
//    template<typename TL, typename TR>                                                         \
//    constexpr auto operator()(TL const& l, TR const & r) const->decltype(_NAME_(l,  r))                 \
//    {  return std::_NAME_(l,  r);   }                                                                       \
//    template<typename TL, typename TR,typename TI>                                                         \
//    constexpr auto operator()(TL const& l, TR const & r,TI const & s) const         \
//    ->decltype(std::_NAME_(traits::get_value(l,s) , traits::get_value( r,s) ))                 \
//    {  return std::_NAME_(traits::get_value(l,s) , traits::get_value( r,s) );   }                                    \
//};
//
//DEF_STD_BINARY_FUNCTION(atan2)
//
//DEF_STD_BINARY_FUNCTION(pow)
//
//#undef DEF_STD_BINARY_FUNCTION
//
//#define DEF_UNARY_FUNCTION(_NAME_)                                                               \
//struct _##_NAME_                                                                             \
//{                                                                                              \
//    template<typename TL >                                                         \
//    constexpr auto operator()(TL const& l ) const->decltype(_NAME_(l ))                 \
//    {  return std::_NAME_(l );   }                                                                       \
//    template<typename TL ,typename TI>                                                         \
//    constexpr auto operator()(TL const& l, TI const & s) const         \
//    ->decltype(std::_NAME_(traits::get_value(l,s)   ))                 \
//    {  return std::_NAME_(traits::get_value(l,s)  );   }                                    \
//    template<typename TL >                                                         \
//    static constexpr auto eval(TL const& l ) ->decltype(_NAME_(l ))                 \
//    {  return std::_NAME_(l );   }                                                                       \
//    template<typename TL ,typename TI>                                                         \
//    static constexpr auto eval(TL const& l, TI const & s)          \
//    ->decltype(std::_NAME_(traits::get_value(l,s)   ))                 \
//    {  return std::_NAME_(traits::get_value(l,s)  );   }                                    \
//};
//
////DEF_UNARY_FUNCTION(fabs)
////DEF_UNARY_FUNCTION(abs)
//DEF_UNARY_FUNCTION(cos)
//
//DEF_UNARY_FUNCTION(acos)
//
//DEF_UNARY_FUNCTION(cosh)
//
//DEF_UNARY_FUNCTION(sin)
//
//DEF_UNARY_FUNCTION(asin)
//
//DEF_UNARY_FUNCTION(sinh)
//
//DEF_UNARY_FUNCTION(tan)
//
//DEF_UNARY_FUNCTION(tanh)
//
//DEF_UNARY_FUNCTION(atan)
//
//DEF_UNARY_FUNCTION(exp)
//
//DEF_UNARY_FUNCTION(log)
//
//DEF_UNARY_FUNCTION(log10)
//
//DEF_UNARY_FUNCTION(sqrt)
//
//DEF_UNARY_FUNCTION(real)
//
//DEF_UNARY_FUNCTION(imag)
//
//#undef DEF_UNARY_FUNCTION
//
//
//struct _pow2
//{
//
//
//    template<typename TL> static constexpr auto eval(TL const &l) DECL_RET_TYPE ((l * l))
//
//    template<typename TL, typename TI>
//    static constexpr auto eval(TL const &l, TI const &s) DECL_RET_TYPE ((_pow2::eval(traits::get_value(l, s))))
//
//    template<typename TL>
//    constexpr TL operator()(TL const &l) const { return _pow2::eval(l); }
//
//    template<typename TL, typename TI>
//    constexpr auto operator()(TL const &l, TI const &s) const DECL_RET_TYPE ((_pow2(traits::get_value(l, s))))
//
//};
//
//struct _identify
//{
//
//
//    template<typename TL>
//    constexpr TL operator()(TL const &l) const { return _identify::eval(l); }
//
//    template<typename TL, typename TI>
//    constexpr auto operator()(TL const &l, TI const &s) const
//    -> decltype(_pow2::eval(traits::get_value(l, s))) { return _identify::eval(traits::get_value(l, s)); }
//
//    template<typename TL>
//    static constexpr TL const &eval(TL const &l) { return l; }
//
//    template<typename TL, typename TI>
//    static constexpr auto eval(TL const &l, TI const &s)
//    -> decltype(_identify::eval(traits::get_value(l, s))) { return _identify::eval(traits::get_value(l, s)); }
//
//};
//
//} // namespace tags
//
//#define _SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _OBJ_, _NAME_)                                                  \
//    template<typename ...T1,typename  T2>  Expression<tags::_NAME_,_OBJ_<T1...>,T2>    \
//    operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
//    {return (Expression<tags::_NAME_,_OBJ_<T1...>,T2> (l,r));}                  \
//
//
//#define _SP_DEFINE_EXPR_BINARY_OPERATOR(_OP_, _OBJ_, _NAME_)                                                  \
//    template<typename ...T1,typename  T2> \
//    Expression<tags::_NAME_,_OBJ_<T1...>,T2>   \
//    operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
//    {return (Expression<tags::_NAME_,_OBJ_<T1...>,T2>(l,r));}                  \
//    template< typename T1,typename ...T2> \
//    Expression< tags::_NAME_,T1,_OBJ_< T2...> >  \
//    operator _OP_(T1 const & l, _OBJ_< T2...>const &r)                    \
//    {return (Expression< tags::_NAME_,T1,_OBJ_< T2...> > (l,r));}                  \
//    template< typename ... T1,typename ...T2> \
//    Expression< tags::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > \
//    operator _OP_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
//    {return (Expression< tags::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > (l,r));}                  \
//
//#define _SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(_OP_, _OBJ_, _NAME_)                                                  \
//        template<typename ...T1,typename  T2> \
//        _OBJ_<AssignmentExpression<tags::_NAME_,_OBJ_<T1...>,T2> >  \
//        operator _OP_(_OBJ_<T1...>   & l,T2 const &r)  \
//        {return (_OBJ_<AssignmentExpression<tags::_NAME_,_OBJ_<T1...>,T2> >  (l,r));}                  \
//
//#define _SP_DEFINE_EXPR_UNARY_OPERATOR(_OP_, _OBJ_, _NAME_)                           \
//        template<typename ...T> \
//        Expression<tags::_NAME_,_OBJ_<T...> >    \
//        operator _OP_(_OBJ_<T...> const &l)  \
//        {return (Expression<tags::_NAME_,_OBJ_<T...> >(l));}   \
//
//#define _SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _OBJ_, _NAME_)                                                  \
//    template<typename ...T1,typename  T2> \
//    BooleanExpression<tags::_NAME_,_OBJ_<T1...>,T2>   \
//    operator _OP_(_OBJ_<T1...> const & l,T2 const &r)  \
//    {return (BooleanExpression<tags::_NAME_,_OBJ_<T1...>,T2>(l,r));}                  \
//    template< typename T1,typename ...T2> \
//    BooleanExpression< tags::_NAME_,T1,_OBJ_< T2...> > \
//    operator _OP_(T1 const & l, _OBJ_< T2...>const &r)                    \
//    {return (BooleanExpression< tags::_NAME_,T1,_OBJ_< T2...> > (l,r));}                  \
//    template< typename ... T1,typename ...T2> \
//    BooleanExpression< tags::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > \
//    operator _OP_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
//    {return (BooleanExpression< tags::_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > (l,r));}                  \
//
//
//#define _SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _OBJ_, _NAME_)                           \
//        template<typename ...T> \
//        BooleanExpression<tags::_NAME_,_OBJ_<T...> >    \
//        operator _OP_(_OBJ_<T...> const &l)  \
//        {return (BooleanExpression<tags::_NAME_,_OBJ_<T...> > (l));}   \
//
//
//#define _SP_DEFINE_EXPR_BINARY_FUNCTION(_NAME_, _OBJ_)                                                  \
//            template<typename ...T1,typename  T2> \
//            Expression<tags::_##_NAME_,_OBJ_<T1...>,T2>    \
//            _NAME_(_OBJ_<T1...> const & l,T2 const &r)  \
//            {return (Expression<tags::_##_NAME_,_OBJ_<T1...>,T2> (l,r));}                  \
//            template< typename T1,typename ...T2> \
//            Expression< tags::_##_NAME_,T1,_OBJ_< T2...> >  \
//            _NAME_(T1 const & l, _OBJ_< T2...>const &r)                    \
//            {return (Expression< tags::_##_NAME_,T1,_OBJ_< T2...> > (l,r));}                  \
//            template< typename ... T1,typename ...T2> \
//            Expression< tags::_##_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > \
//            _NAME_(_OBJ_< T1...> const & l,_OBJ_< T2...>  const &r)                    \
//            {return (Expression< tags::_##_NAME_,_OBJ_< T1...>,_OBJ_< T2...> > (l,r));}                  \
//
//
//#define _SP_DEFINE_EXPR_UNARY_FUNCTION(_NAME_, _OBJ_)                           \
//        template<typename ...T> Expression<tags::_##_NAME_,_OBJ_<T ...> >  \
//        _NAME_(_OBJ_<T ...> const &r)  \
//        {return (Expression<tags::_##_NAME_,_OBJ_<T ...> > (r));}   \
//
//
//#define  DEFINE_EXPRESSION_TEMPLATE_BASIC_ALGEBRA(_CONCEPT_)          \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(+, _CONCEPT_, plus)                   \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(-, _CONCEPT_, minus)                  \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(*, _CONCEPT_, multiplies)             \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(/, _CONCEPT_, divides)                \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(%, _CONCEPT_, modulus)                \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(^, _CONCEPT_, bitwise_xor)            \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(&, _CONCEPT_, bitwise_and)            \
//_SP_DEFINE_EXPR_BINARY_OPERATOR(|, _CONCEPT_, bitwise_or)             \
//_SP_DEFINE_EXPR_UNARY_OPERATOR(~, _CONCEPT_, bitwise_not)             \
//_SP_DEFINE_EXPR_UNARY_OPERATOR(+, _CONCEPT_, unary_plus)              \
//_SP_DEFINE_EXPR_UNARY_OPERATOR(-, _CONCEPT_, negate)                  \
//_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(<<, _CONCEPT_, shift_left)      \
//_SP_DEFINE_EXPR_BINARY_RIGHT_OPERATOR(>>  , _CONCEPT_, shift_right)     \
//_SP_DEFINE_EXPR_BINARY_FUNCTION(atan2, _CONCEPT_)                      \
//_SP_DEFINE_EXPR_BINARY_FUNCTION(pow, _CONCEPT_)                      \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(cos, _CONCEPT_)                        \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(acos, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(cosh, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(sin, _CONCEPT_)                        \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(asin, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(sinh, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(tan, _CONCEPT_)                        \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(tanh, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(atan, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(exp, _CONCEPT_)                        \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(log, _CONCEPT_)                        \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(log10, _CONCEPT_)                      \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(sqrt, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(real, _CONCEPT_)                       \
//_SP_DEFINE_EXPR_UNARY_FUNCTION(imag, _CONCEPT_)                       \


//DEFINE_EXPRESSION_TEMPLATE_BASIC_ALGEBRA(Expression)

//_SP_DEFINE_EXPR_UNARY_BOOLEAN_OPERATOR(!, _CONCEPT_, logical_not)     \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(&&, _CONCEPT_, logical_and)   \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(||, _CONCEPT_, logical_or)    \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(!=, _CONCEPT_, not_equal_to)  \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(==, _CONCEPT_, equal_to)      \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<, _CONCEPT_, less)           \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>, _CONCEPT_, greater)        \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(<=, _CONCEPT_, less_equal)    \
//_SP_DEFINE_EXPR_BINARY_BOOLEAN_OPERATOR(>=, _CONCEPT_, greater_equal) \

//_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(+=,_CONCEPT_, plus_assign)        \
//_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(-=,_CONCEPT_, minus_assign)       \
//_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(*=,_CONCEPT_, multiplies_assign)  \
//_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(/=,_CONCEPT_, divides_assign)     \
//_SP_DEFINE_EXPR_ASSIGNMENT_OPERATOR(%=,_CONCEPT_, modulus_assign)     \



}}//namespace simpla{namespace algebra{
#endif //SIMPLA_PRIMARYOPS_H
