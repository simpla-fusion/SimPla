/*
 * test_visitor.cpp
 *
 *  Created on: 2013年12月22日
 *      Author: salmon
 */
#include <tuple>
#include <iostream>
struct VistorBase
{
	VistorBase()
	{
	}
	virtual ~VistorBase()
	{
	}
	virtual void visit(void *obj)=0;
};

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

template<typename T, typename ... Args>
struct NextTimeStepVistor: public VistorBase
{
	std::tuple<Args ...> args_;

	NextTimeStepVistor(Args const & ... args)
			: args_(std::make_tuple(std::forward<Args const &>(args)...))
	{
	}
	~NextTimeStepVistor()
	{
	}
	void visit(void *obj)
	{
		callFunc(obj, typename GenSeq<sizeof...(Args)>::type());
	}

	template<int ...S>
	void callFunc(void* obj,Seq<S...>)
	{
		reinterpret_cast<T*>(obj)->NextTimeStep(std::get<S>(args_) ...);
	}

};
template<typename T, typename ...Args>
VistorBase * CreateVistor(Args const &... args)
{
	return (new NextTimeStepVistor<T, Args...>(std::forward<Args const &>(args)...));
}

struct Base
{
	virtual ~Base()
	{

	}
	virtual void accept(VistorBase *)=0;
};
struct Foo1: public Base
{
	void accept(VistorBase * visitor)
	{
		visitor->visit(this);
	}

	void NextTimeStep(int a, int b)
	{
		std::cout << "This is Foo1 " << a << "     " << b << std::endl;
	}
};
struct Foo2: public Base
{
	void accept(VistorBase * visitor)
	{
		visitor->visit(this);
	}

	void NextTimeStep(int a, int b, int c)
	{
		std::cout << "This is Foo2 " << a << "     " << b << std::endl;
	}
};
int main(int argc, char **argv)
{
	Base * f1 = dynamic_cast<Base*>(new Foo1);
	Base * f2 = dynamic_cast<Base*>(new Foo2);
	VistorBase *v1 = CreateVistor<Foo1>(5, 6);
	VistorBase *v2 = CreateVistor<Foo2>(5, 6, 3);
	f1->accept(v1);
	f2->accept(v2);

}

