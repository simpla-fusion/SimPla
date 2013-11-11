#include <tuple>
#include <iostream>
#include <functional>

template<int N>
struct Int2Type
{
	static const int NUM = N;
};
struct C
{
	template<typename ... Args>
	void foo(int a, Args const & ... args) const
	{
		std::cout << sizeof...(args) << std::endl;
	}
};

template<typename T>
T foo(T const &)
{
	return T();
}
template<typename ...Args>
void foo2(C const &c, Args const & ...args)
{
	std::mem_fn(&C::foo)(c, args...);
}

int main(void)
{
	C c;
decltype(MakeCache(args,0)...)...
foo2(c, 1, 3, 4, 5);

}
