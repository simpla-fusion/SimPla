/**
 * @file macro.h
 *
 * @date 2015-6-12
 * @author salmon
 */

#ifndef CORE_toolbox_MACRO_H_
#define CORE_toolbox_MACRO_H_

#pragma warning( disable : 1334)

/**
 * Count the number of arguments passed to MACRO, very carefully
 * tiptoeing around an MSVC bug where it improperly expands __VA_ARGS__ as a
 * single token in argument lists.  See these URLs for details:
 *
 *   http://stackoverflow.com/questions/9183993/msvc-variadic-macro-expansion/9338429#9338429
 *   http://connect.microsoft.com/VisualStudio/feedback/details/380090/variadic-macro-replacement
 *   http://cplusplus.co.il/2010/07/17/variadic-macro-to-count-number-of-arguments/#comment-644
 */
#define COUNT_MACRO_ARGS_IMPL2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, count, ...) count
#define COUNT_MACRO_ARGS_IMPL(args) COUNT_MACRO_ARGS_IMPL2 args
#define COUNT_MACRO_ARGS(...) COUNT_MACRO_ARGS_IMPL((__VA_ARGS__,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))


#define TEMPLATE_DISPATCH(_FUN_NAME_, _PREFIX_, _SUFFIX_)                                            \
    template<typename ..._ARGS> _PREFIX_ void _dispatch_##_FUN_NAME_(_ARGS &&... args) _SUFFIX_{ };        \
    template<typename _FIRST, typename ..._OTHERS, typename ..._ARGS>            \
     _PREFIX_  void _dispatch_##_FUN_NAME_(_ARGS &&... args) _SUFFIX_                                        \
    {                                                                            \
        _FIRST::_FUN_NAME_(std::forward<_ARGS>(args)...);                        \
        _dispatch_##_FUN_NAME_<_OTHERS...>(std::forward<_ARGS>(args)...);                 \
    };                                                                           \

#define TEMPLATE_DISPATCH_DEFAULT(_FUN_NAME_)    TEMPLATE_DISPATCH( _FUN_NAME_ , , )


//**********************************
// modified from google test
#define SIMPLA_DISALLOW_ASSIGN(_TYPE_)   void operator=(_TYPE_ const &)=delete;

#define SIMPLA_DISALLOW_COPY_AND_ASSIGN(_TYPE_)   _TYPE_(_TYPE_ const &)=delete; SIMPLA_DISALLOW_ASSIGN(_TYPE_)


#if __cplusplus < 201402L
#   define AUTO_RETURN(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}
#else
//#   define AUTO_RETURN(_EXPR_)  {return  (_EXPR_);}
#endif
//**********************************
#endif /* CORE_toolbox_MACRO_H_ */
