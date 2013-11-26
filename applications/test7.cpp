#include "../src/fetl/primitives.h"

#include <iostream>

using namespace simpla;

struct B
{
	virtual ~B()
	{
	}

	virtual void Foo(Int2Type<0>)=0;
	virtual void Foo(Int2Type<1>)=0;

};

class A: public B
{
	template<int N> void Foo(Int2Type<N>)
	{
		std::cout << N << std::endl;
	}

	void Foo(Int2Type<0>)
	{
		std::cout << 0 << std::endl;
	}
	void Foo(Int2Type<1>)
	{
		std::cout << 1 << std::endl;
	}
};
int main(int argc, char** argv)
{
	A a;
	B* b = dynamic_cast<B*>(&a);

	b->Foo(Int2Type<0>());
	b->Foo(Int2Type<1>());
	a.Foo(Int2Type<1>());
}

// vim:et
