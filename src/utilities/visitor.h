/*
 * visitor.h
 *
 *  Created on: 2014年4月23日
 *      Author: salmon
 */

#ifndef VISITOR_H_
#define VISITOR_H_

namespace simpla
{

//******************************************************************************************************
// Generic Visitor
//******************************************************************************************************

struct VisitorBase
{

	VisitorBase()
	{
	}
	virtual ~VisitorBase()
	{
	}
	virtual void Visit(void*) const=0;

	virtual std::string GetTypeAsString_() const
	{
		return "Custom";
	}

	std::string GetTypeAsString() const
	{
		return GetTypeAsString_();
	}
};

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

//
//struct AcceptorBase
//{
//	virtual ~AcceptorBase()
//	{
//
//	}
//	virtual void accept(std::shared_ptr<VisitorBase> visitor)
//	{
//		visitor->visit(this);
//	}
//	virtual bool CheckType(std::type_info const &)
//	{
//		return false;
//	}
//};
//
//template<typename T, typename ...Args>
//struct Visitor: public VisitorBase
//{
//	std::string name_;
//public:
//
//	typedef std::tuple<Args...> args_tuple_type;
//	std::tuple<Args...> args_;
//
//	Visitor(std::string const &name, Args ... args)
//			: name_(name), args_(std::make_tuple(args...))
//	{
//	}
//	Visitor(Args ... args)
//			: name_(""), args_(std::make_tuple(args...))
//	{
//	}
//	~Visitor()
//	{
//	}
//	inline const std::string& GetName() const
//	{
//		return name_;
//	}
//	void visit(AcceptorBase* obj)
//	{
//		if (obj->CheckType(typeid(T)))
//		{
//			reinterpret_cast<T*>(obj)->template accept(*this);
//		}
//		else
//		{
//			ERROR << "acceptor type mismatch";
//		}
//	}
//
//	template<typename TFUN>
//	inline void excute(TFUN const & f)
//	{
//		callFunc(f, typename GenSeq<sizeof...(Args)>::type() );
//	}
//
//private:
//// Unpack tuple to args...
////@ref http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
//
//	template<int...>
//	struct Seq
//	{};
//
//	template<int N, int ...S>
//	struct GenSeq: GenSeq<N - 1, N - 1, S...>
//	{
//	};
//
//	template<int ...S>
//	struct GenSeq<0, S...>
//	{
//		typedef Seq<S...> type;
//	};
//
//	template<typename TFUN, int ...S>
//	inline void callFunc(TFUN const & fun, Seq<S...>)
//	{
//		fun(std::get<S>(args_) ...);
//	}
//
//};
//
//template<typename T, typename ...Args>
//std::shared_ptr<VisitorBase> CreateVisitor(std::string const & name, Args ...args)
//{
//	return std::dynamic_pointer_cast<VisitorBase>(std::shared_ptr<Visitor<T, Args...>>(
//
//	new Visitor<T, Args...>(name, std::forward<Args &>(args)...)));
//}
}// namespace simpla

#endif /* VISITOR_H_ */
