/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include <iostream>

template<typename T> struct U
{
	static constexpr size_t value = 0;
};

template<typename T, size_t N> struct U<T[N]>
{
	static constexpr size_t value = N;
};
int main(int argc, char **argv)
{
	int a[4][5];

	std::cout<<decltype(a[0])
}

