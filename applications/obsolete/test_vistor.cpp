/*
 * test_visitor.cpp
 *
 *  Created on: 2013年12月22日
 *      Author: salmon
 */
#include <tuple>
#include <iostream>
#include <memory>

struct AcceptorBase;
struct VisitorBase
{
	std::map<std::pair<size_t, size_t>, std::function<void(AcceptorBase*, VisitorBase*)> > callmap_;

	VisitorBase()
	{
	}
	virtual ~VisitorBase()
	{
	}
	virtual void visit(AcceptorBase* obj) const=0;

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

	size_t type_id_hash;
};

template<typename ...Args>
struct ArgsPack: public VisitorBase
{
	std::string name_;
	std::tuple<Args...> args_;

	ArgsPack(std::string const name, Args ...args) :
			name_(name), args_(std::make_tuple(args...))
	{
	}
};

template<typename T, typename ...Args>
struct Visitor: public VisitorBase
{
	std::string name_;
	std::tuple<Args...> args_;
public:

	Visitor(std::string const name, Args ...args) :
			name_(name), args_(std::make_tuple(args...))
	{
	}
	~Visitor()
	{
	}

	void visit(T* obj) const
	{
		obj->accept(this);

	}

	template<typename TFUN>
	void excute(TFUN const fun)
	{
		callFunc(fun, typename GenSeq<sizeof...(Args)>::type());
	}

private:
// Unpack tuple to args...
//@ref http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer

	std::tuple<Args...> args_;
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

template<typename ...Args>
std::shared_ptr<VisitorBase> CreateVisitor(std::string const & name, Args ...args)
{
	return std::dynamic_pointer_cast<VisitorBase>(
			std::shared_ptr<Visitor<Args...>>(new Visitor<Args...>(name, std::forward<Args &>(args)...)));
}

struct Foo1: public AcceptorBase
{
	typedef Foo1 this_type;

	virtual bool CheckType(std::type_info const &t_info)
	{
		return typeid(this_type) == t_info;
	}

	void accept(VisitorBase const &visitor)
	{
		visitor.visit(this);
	}

	template<typename ...Args>
	void accept(Visitor<this_type, Args...> &visitor)
	{
		visitor.excute([this](Args ... args)
		{	this->Command(std::forward<Args>(args)...);});
	}

	void Command2(std::string const & s)
	{
		std::cout << "This is Foo1::Command2(string). args=" << s << std::endl;
	}

	void Command(int a, int b)
	{
		std::cout << "This is Foo1::Command(int,int). args=" << a << "     " << b << std::endl;
	}

	template<typename ... Args>
	void Command(Args const & ...args)
	{
		std::cout << "This is Foo1::Command(args...). args=";

		Print(args...);

		std::cout << std::endl;
	}

	void Print()
	{
	}

	template<typename T, typename ... Others>
	void Print(T const &v, Others const & ... others)
	{
		std::cout << v << " ";
		Print(std::forward<Others const &>(others )...);
	}

};

int main(int argc, char **argv)
{
	AcceptorBase * f1 = dynamic_cast<AcceptorBase*>(new Foo1);
	auto v1 = CreateVisitor("Command1", 5, 6);
	auto v2 = CreateVisitor("Command2", "hello world");
	auto v3 = CreateVisitor("Command3", 5, 6, 3);
	f1->accept(v1);
	f1->accept(v2);
	f1->accept(v3);

	delete f1;

}

