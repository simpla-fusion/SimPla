/*
 * test1.cpp
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#include <iostream>

namespace Bar
{

template<typename T>
class Foo
{
public:
	template<typename U>
	friend std::ostream& operator<<(std::ostream& os, const Foo<U>& foo);

	Foo(T x) :
			_x(x)
	{
	}

	T x() const
	{
		return _x;
	}

private:

	T _x;
};

template<typename T>
std::ostream& operator <<(std::ostream& os, const Foo<T>& foo)
{
	os << foo._x;
	return os;
}

}
int main(int argc, char **argv)
{

	Bar::Foo<int> foo(5);
	std::cout << foo << std::endl;
}

