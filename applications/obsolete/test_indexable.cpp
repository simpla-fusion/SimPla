#include <complex>

#include <typeinfo>

#include <cstddef>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include <array>
#include <iostream>

template<class T, typename TI = int>
struct is_indexable
{
	template<typename T1, typename T2>
	static auto check_index(T1 const& u,
			T2 const &s) -> decltype(const_cast<typename std::remove_cv<T1>::type &>(u)[s])
	{
	}

	template<typename T1, typename T2>
	static auto check_const_index_only(T1 const &u,
			T2 const &s) -> decltype(u[s])
	{
	}

	static std::false_type check_index(...)
	{
		return std::false_type();
	}
	static std::false_type check_const_index_only(...)
	{
		return std::false_type();
	}
public:

//	typedef decltype(
//			check_const_index(std::declval<T>(),
//					std::declval<TI>())) const_result_type;

	typedef decltype(
			check_index((std::declval<T>()),
					std::declval<TI>())) result_type;

	typedef decltype(
			check_const_index_only((std::declval<T>()),
					std::declval<TI>())) const_result_type;

	static const bool value =
			!(std::is_same<result_type, std::false_type>::value);

	static const bool has_const_ref = !(std::is_same<const_result_type,
			std::false_type>::value);

	static const bool has_non_const_ref = value
			&& (!std::is_const<result_type>::value);

//			!(std::is_same<const_result_type, std::false_type>::value)
	;
};

class Foo
{
public:
};
class Foo1
{
	int v;
public:
	Foo1(Foo1 const &) = delete;
	int & operator[](size_t)
	{
		return v;
	}
};
class Foo2
{
	int v;
public:
	Foo2(Foo const &) = delete;

	int const & operator[](size_t) const
	{
		return v;
	}
};

int main()
{
	std::cout << " std::string is indexable. " << std::boolalpha
			<< is_indexable<std::string>::value << std::endl;

	std::cout << " double* is indexable. " << std::boolalpha
			<< is_indexable<double *>::value << std::endl;

	std::cout << " double is indexable. " << std::boolalpha
			<< is_indexable<double,int>::value << std::endl;

	std::cout << " std::map<int, std::string> is indexable for int. "
			<< std::boolalpha
			<< is_indexable<std::map<int, std::string>, int>::value
			<< std::endl;

	std::cout << " std::map<int, std::string> is indexable for double. "
			<< std::boolalpha
			<< is_indexable<std::map<int, std::string>, double>::value
			<< std::endl;

	std::cout << " std::map<int, std::string> is indexable for std::string. "
			<< std::boolalpha
			<< is_indexable<std::map<int, std::string>, std::string>::value
			<< std::endl;
	std::cout << " std::complex<double> is indexable for int. "
			<< std::boolalpha << is_indexable<std::complex<double>, int>::value
			<< std::endl;

	std::cout << " Foo is indexable. " << std::boolalpha
			<< is_indexable<Foo>::value << std::endl;

	std::cout << " Foo1 is indexable. " << std::boolalpha
			<< is_indexable<Foo1>::value << "  "
			<< " Foo1 has constant index operator. " << std::boolalpha
			<< is_indexable<Foo1>::has_const_ref
			<< " Foo1 has non-constant index operator. " << std::boolalpha
			<< is_indexable<Foo1>::has_non_const_ref << std::endl;

	std::cout << " Foo2 is indexable. " << std::boolalpha
			<< is_indexable<Foo2>::value << "  "
			<< " Foo2 has constant index operator. " << std::boolalpha
			<< is_indexable<Foo2>::has_const_ref
			<< " Foo2 has non-constant index operator. " << std::boolalpha
			<< is_indexable<Foo2>::has_non_const_ref << std::endl;
}
