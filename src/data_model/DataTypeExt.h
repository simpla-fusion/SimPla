/**
 * @file DataTypeExt.h
 *
 *  Created on: 2015-6-10
 *      Author: salmon
 */

#ifndef CORE_DATASET_DATATYPE_EXT_H_
#define CORE_DATASET_DATATYPE_EXT_H_


/*
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

/* Pick the right helper macro to invoke. */
#define SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_, _N0_)  ::simpla::traits::primary_type_t<_T0_> _N0_;

#define SP_DEFINE_STRUCT_MEMBER_HELPER4(_T0_, _N0_, _T1_, _N1_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER2(_T1_,_N1_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER6(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER4(_T1_,_N1_,_T2_,_N2_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER8(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER10(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER12(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER14(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER16(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_, _T7_, _N7_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER18(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_, _T7_, _N7_, _T8_, _N8_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER16(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
#define SP_DEFINE_STRUCT_MEMBER_HELPER20(_T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_, _T7_, _N7_, _T8_, _N8_, _T9_, _N9_) SP_DEFINE_STRUCT_MEMBER_HELPER2(_T0_,_N0_) \
      SP_DEFINE_STRUCT_MEMBER_HELPER18(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_,_T9_,_N9_)


#define SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER1(count) SP_DEFINE_STRUCT_MEMBER_HELPER##count
#define SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER(count) SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER1(count)
#define SP_DEFINE_STRUCT_MEMBER(...) SP_DEFINE_STRUCT_MEMBER_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

//#define SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) _N0_
//#define SP_PARTICLE_GET_NAME_HELPER4(_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER2(_T1_,_N1_)
//#define SP_PARTICLE_GET_NAME_HELPER6(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER4(_T1_,_N1_,_T2_,_N2_)
//#define SP_PARTICLE_GET_NAME_HELPER8(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
//#define SP_PARTICLE_GET_NAME_HELPER10(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
//#define SP_PARTICLE_GET_NAME_HELPER12(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
//#define SP_PARTICLE_GET_NAME_HELPER14(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
//#define SP_PARTICLE_GET_NAME_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
//#define SP_PARTICLE_GET_NAME_HELPER18(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER16(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
//
//#define SP_PARTICLE_GET_NAME_CHOOSE_HELPER1(count) SP_PARTICLE_GET_NAME_HELPER##count
//#define SP_PARTICLE_GET_NAME_CHOOSE_HELPER(count) SP_PARTICLE_GET_NAME_CHOOSE_HELPER1(count)
//#define SP_PARTICLE_GET_NAME(...) SP_PARTICLE_GET_NAME_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

#define SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_, _T0_, _N0_) d_type.push_back(::simpla::data_model::DataType::template create<::simpla::traits::primary_type_t<_T0_> >(), #_N0_, offsetof(_S_NAME_, _N0_));
#define SP_DEFINE_STRUCT_DESC_HELPER4(_S_NAME_, _T0_, _N0_, _T1_, _N1_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T1_,_N1_)
#define SP_DEFINE_STRUCT_DESC_HELPER6(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_)  SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER4(_S_NAME_,_T1_,_N1_,_T2_,_N2_)
#define SP_DEFINE_STRUCT_DESC_HELPER8(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER6(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_DEFINE_STRUCT_DESC_HELPER10(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER8(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_DEFINE_STRUCT_DESC_HELPER12(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER10(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_DEFINE_STRUCT_DESC_HELPER14(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER12(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_DEFINE_STRUCT_DESC_HELPER16(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_, _T7_, _N7_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_DEFINE_STRUCT_DESC_HELPER18(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_, _T7_, _N7_, _T8_, _N8_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER16(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
#define SP_DEFINE_STRUCT_DESC_HELPER20(_S_NAME_, _T0_, _N0_, _T1_, _N1_, _T2_, _N2_, _T3_, _N3_, _T4_, _N4_, _T5_, _N5_, _T6_, _N6_, _T7_, _N7_, _T8_, _N8_, _T9_, _N9_) SP_DEFINE_STRUCT_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
      SP_DEFINE_STRUCT_DESC_HELPER18(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_,_T9_,_N9_)

#define SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER1(count) SP_DEFINE_STRUCT_DESC_HELPER##count
#define SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER(count) SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER1(count)
#define SP_DEFINE_STRUCT_DESC(_S_NAME_, ...) SP_DEFINE_STRUCT_DESC_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S_NAME_,__VA_ARGS__)


#define SP_DEFINE_STRUCT_DATATYPE2(_S_NAME_, ...) \
   static ::simpla::data_model::DataType data_type()   \
    {                                                   \
        ::simpla::data_model::DataType d_type(typeid(_S_NAME_),sizeof(_S_NAME_),0,nullptr,__STRING(_S_NAME_));                                \
        SP_DEFINE_STRUCT_DESC(_S_NAME_, __VA_ARGS__);   \
        return std::move(d_type);                       \
    }                                                   \


#ifdef __cplusplus

#   define SP_DEFINE_STRUCT_TYPE_DESC(_S_NAME_, ...)                    \
   static constexpr bool is_self_describing=true;                   \
   static std::string  name(){return __STRING(_S_NAME_);}          \
   static ::simpla::data_model::DataType data_type()                \
    {                                                               \
        ::simpla::data_model::DataType d_type(typeid(_S_NAME_),     \
       sizeof(_S_NAME_),0,nullptr,__STRING(_S_NAME_));              \
        SP_DEFINE_STRUCT_DESC(_S_NAME_, __VA_ARGS__);               \
        return std::move(d_type);                                   \
    }
#else
#   define SP_DEFINE_STRUCT_TYPE_DESC(_S_NAME_, ...)
#endif


#define SP_DEFINE_STRUCT(_S_NAME_, ...)                       \
struct _S_NAME_{SP_DEFINE_STRUCT_MEMBER(__VA_ARGS__)        \
SP_DEFINE_STRUCT_TYPE_DESC(_S_NAME_, __VA_ARGS__) };             \



#endif /* CORE_DATASET_DATATYPE_EXT_H_ */
