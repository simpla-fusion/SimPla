#include <iostream>

template<typename T, int I>
class Foo
{
public:

	double a[4];

	constexpr double foo() const
	{
		return a[0] + a[1] + b;
	}

	double foo2()
	{
		double c = 100 + b;
		return c;
	}
	static constexpr int b = 10;
};

int main(int argc, char const *argv[])
{
	/* code */
	Foo<int, 1> fi;

	std::cout << fi.foo() << std::endl;
	std::cout << fi.foo2() << std::endl;
	return 0;
}
