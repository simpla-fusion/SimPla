/*
 * type_utilites.h
 *
 *  Created on: 2013年12月14日
 *      Author: salmon
 */

#ifndef TYPE_UTILITES_H_
#define TYPE_UTILITES_H_

#include "log.h"

namespace simpla
{

#include <tuple>
template<int N> struct Int2Type
{
	static const int value = N;
};

struct NullType;

struct EmptyType
{
};

enum CONST_NUMBER
{
	ZERO = 0, ONE = 1, TWO = 2, THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8, NINE = 9
};

#define HAS_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(std::declval<U>()._NAME_() )>::type;                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(std::declval<U>()._NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
};

#define HAS_STATIC_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_static_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(U::_NAME_() )>::type;                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(U::_NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
};

#define HAS_FUNCTION(_NAME_)                                                                   \
template< typename ..._Args>                                                                \
struct has_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(_NAME_() )>::type;                                                       \
                                                                                                      \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(_NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(0), no>::value;                     \
};

#define HAS_OPERATOR(_NAME_,_OP_)                                                                   \
template<typename _T, typename ... _Args>                                                                \
struct has_operator_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename _U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(std::declval<_U>().operator _OP_() )>::type;                                                       \
                                                                                                      \
	template<typename _U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(std::declval<_U>().operator _OP_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;                     \
};

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}

#define ENABLE_IF_DECL_RET_TYPE(_COND_,_EXPR_) \
        ->typename std::enable_if<_COND_,decltype((_EXPR_))>::type {return (_EXPR_);}

#define COND_DECL_RET_TYPE(_COND_,_EXPR_,_FAILSAFE_) \
        ->typename std::conditional<_COND_,decltype((_EXPR_)),_FAILSAFE_>::type {return (_EXPR_);}

//template<typename T>
//struct remove_const_reference
//{
//	typedef typename std::remove_const<typename std::remove_reference<T>::type>::type type;
//};
//template<typename T>
//struct is_storage_type
//{
//	static constexpr bool value = true;
//};
//template<typename T>
//struct is_storage_type<T*>
//{
//	static constexpr bool value = true;
//};
//template<typename T>
//struct ReferenceTraits // obsolete
//{
//	typedef typename remove_const_reference<T>::type TL;
//	typedef typename std::conditional<is_storage_type<TL>::value, TL &, TL>::type type;
//};
//
//template<typename T>
//struct ConstReferenceTraits // obsolete
//{
//	typedef typename remove_const_reference<T>::type TL;
//	typedef typename std::conditional<is_storage_type<TL>::value, TL const &, const TL>::type type;
//};

template<typename T>
struct can_not_reference
{
	static constexpr bool value = false;
};

template<typename T>
struct StorageTraits
{
	static constexpr bool not_refercne = std::is_pointer<T>::value

	|| std::is_reference<T>::value

	|| std::is_scalar<T>::value

	|| can_not_reference<T>::value;

	typedef typename std::conditional<not_refercne, T, T&>::type type;

	typedef typename std::conditional<not_refercne, T, const T&>::type const_reference;

	typedef typename std::conditional<not_refercne, T, T&>::type reference;

};

template<class T, typename TI = int>
class is_indexable
{
	HAS_OPERATOR(index, []);
public:
	static const bool value = has_operator_index<T, TI>::value;

};

// @ref http://stackoverflow.com/questions/3913503/metaprogram-for-bit-counting
template<int N>
struct CountBits
{
	static const unsigned int n = CountBits<N / 2>::n + 1;
};

template<>
struct CountBits<0>
{
	static const unsigned int n = 0;
};

unsigned int count_bits(unsigned long s)
{
	unsigned int n = 0;
	while (s > 0)
	{
		++n;
		s = s >> 1;
	}
	return n;
}
//******************************************************************************************************
// Generic Visitor
//******************************************************************************************************

struct AcceptorBase;

/***
 *  \brief Vistor<T>
 *
 *  Double Visitor pattern :
 *  purpose: pass variadic parameters to acceptor
 *  visitor visit acceptor twice, first get acceptor type , second get parameters type .
 *  \code
 *   struct Foo1: public AcceptorBase
 *   {
 *   	typedef Foo1 this_type;
 *
 *   	virtual bool CheckType(std::type_info const &t_info)
 *   	{
 *   		return typeid(this_type) == t_info;
 *   	}
 *
 *   	template<typename ...Args>
 *   	void accept(Visitor<this_type, Args...> &visitor)
 *   	{
 *   		visitor.excute([this](Args ... args)
 *   		{	this->Command(std::forward<Args>(args)...);});
 *   	}
 *   	void accept(Visitor<this_type, const char *> &visitor)
 *   	{
 *   		if (visitor.GetName() == "Command2")
 *   		{
 *   			visitor.excute([this](std::string const & args)
 *   			{	this->Command2(args);});
 *   		}
 *   		else
 *   		{
 *   			std::cout << "Unknown function name!" << std::endl;
 *   		}
 *   	}
 *
 *   	void Command2(std::string const & s)
 *   	{
 *   		std::cout << "This is Foo1::Command2(string). args=" << s << std::endl;
 *   	}
 *
 *   	void Command(int a, int b)
 *   	{
 *   		std::cout << "This is Foo1::Command(int,int). args=" << a << "     " << b << std::endl;
 *   	}
 *
 *   	template<typename ... Args>
 *   	void Command(Args const & ...args)
 *   	{
 *   		std::cout << "This is Foo1::Command(args...). args=";
 *
 *   		Print(args...);
 *
 *   		std::cout << std::endl;
 *   	}
 *
 *   	void Print()
 *   	{
 *   	}
 *
 *   	template<typename T, typename ... Others>
 *   	void Print(T const &v, Others const & ... others)
 *   	{
 *   		std::cout << v << " ";
 *   		Print(std::forward<Others const &>(others )...);
 *   	}
 *
 *   };
 *
 *   int main(int argc, char **argv)
 *   {
 *   	AcceptorBase * f1 = dynamic_cast<AcceptorBase*>(new Foo1);
 *   	auto v1 = CreateVisitor<Foo1>("Command1", 5, 6);
 *   	auto v2 = CreateVisitor<Foo1>("Command2", "hello world");
 *   	auto v3 = CreateVisitor<Foo1>("Command3", 5, 6, 3);
 *   	f1->accept(v1);
 *   	f1->accept(v2);
 *   	f1->accept(v3);
 *
 *   	delete f1;
 *
 *   }
 *  \endcode
 *
 */

struct VisitorBase
{

	VisitorBase()
	{
	}
	virtual ~VisitorBase()
	{
	}
	virtual void visit(AcceptorBase* obj)=0;

};

struct AcceptorBase
{
	virtual ~AcceptorBase()
	{

	}
	virtual void accept(std::shared_ptr<VisitorBase> visitor)
	{
		visitor->visit(this);
	}
	virtual bool CheckType(std::type_info const &)
	{
		return false;
	}
};

template<typename T, typename ...Args>
struct Visitor: public VisitorBase
{
	std::string name_;
public:

	typedef std::tuple<Args...> args_tuple_type;
	std::tuple<Args...> args_;

	Visitor(std::string const &name, Args ... args)
			: name_(name), args_(std::make_tuple(args...))
	{
	}
	Visitor(Args ... args)
			: name_(""), args_(std::make_tuple(args...))
	{
	}
	~Visitor()
	{
	}
	inline const std::string& GetName() const
	{
		return name_;
	}
	void visit(AcceptorBase* obj)
	{
		if (obj->CheckType(typeid(T)))
		{
			reinterpret_cast<T*>(obj)->template accept(*this);
		}
		else
		{
			ERROR << "acceptor type mismatch";
		}
	}

	template<typename TFUN>
	inline void excute(TFUN const & f)
	{
		callFunc(f, typename GenSeq<sizeof...(Args)>::type() );
	}

private:
// Unpack tuple to args...
//@ref http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer

	template<int...>
	struct Seq
	{};

	template<int N, int ...S>
	struct GenSeq: GenSeq<N - 1, N - 1, S...>
	{
	};

	template<int ...S>
	struct GenSeq<0, S...>
	{
		typedef Seq<S...> type;
	};

	template<typename TFUN, int ...S>
	inline void callFunc(TFUN const & fun, Seq<S...>)
	{
		fun(std::get<S>(args_) ...);
	}

};

template<typename T, typename ...Args>
std::shared_ptr<VisitorBase> CreateVisitor(std::string const & name, Args ...args)
{
	return std::dynamic_pointer_cast<VisitorBase>(std::shared_ptr<Visitor<T, Args...>>(

	new Visitor<T, Args...>(name, std::forward<Args &>(args)...)));
}

template<typename T> inline T* PointerTo(T & v)
{
	return &v;
}

template<typename T> inline T* PointerTo(T * v)
{
	return v;
}

} // namespace simpla

#endif /* TYPE_UTILITES_H_ */
