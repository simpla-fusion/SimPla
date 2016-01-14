/**
 * @file visitor.h
 *
 *  created on: 2014-4-23
 *      Author: salmon
 */

#ifndef VISITOR_H_
#define VISITOR_H_

#include <string>
#include <memory>
#include "../utilities/log.h"
#include "../type_traits.h"

namespace simpla
{
/**
 * @ingroup  design_pattern
 *
 * @addtogroup vistor Vistor
 * @{
 **/
/**
 *  \brief  Generic Visitor
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
 *   	void filter(Visitor<this_type, Args...> &visitor)
 *   	{
 *   		visitor.excute([this](Args ... args)
 *   		{	this->Command(std::forward<Args>(args)...);});
 *   	}
 *   	void filter(Visitor<this_type, const char *> &visitor)
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
 *   	void Command(int a,  unsigned int  b)
 *   	{
 *   		std::cout << "This is Foo1::Command(unsigned int ,int). args=" << a << "     " << b << std::endl;
 *   	}
 *
 *   	template<typename ... Args>
 *   	void Command(Args const & ...args)
 *   	{
 *   		std::cout << "This is Foo1::Command(args...). args=";
 *
 *   		print(args...);
 *
 *   		std::cout << std::endl;
 *   	}
 *
 *   	void print()
 *   	{
 *   	}
 *
 *   	template<typename T, typename ... Others>
 *   	void print(T const &v, Others && ... others)
 *   	{
 *   		std::cout << v << " ";
 *   		print(std::forward<Others >(others )...);
 *   	}
 *
 *   };
 *
 *    unsigned int  main(int argc, char **argv)
 *   {
 *   	AcceptorBase * f1 = dynamic_cast<AcceptorBase*>(new Foo1);
 *   	auto v1 = createVisitor<Foo1>("Command1", 5, 6);
 *   	auto v2 = createVisitor<Foo1>("Command2", "hello world");
 *   	auto v3 = createVisitor<Foo1>("Command3", 5, 6, 3);
 *   	f1->filter(v1);
 *   	f1->filter(v2);
 *   	f1->filter(v3);
 *
 *   	delete f1;
 *
 *   }
 *  \endcode
 *
 */
struct VisitorBase
{

    VisitorBase() { }

    virtual ~VisitorBase() { }

    void visit(void *p) const { visit_(p); }

    virtual std::string get_type_as_string() const { return "Custom"; }

private:

    virtual void visit_(void *) const = 0;
};


struct AcceptorBase
{
    virtual ~AcceptorBase()
    {

    }

    virtual void accept(VisitorBase &visitor)
    {
        visitor.visit(this);
    }

    virtual void accept(VisitorBase const &visitor)
    {
        visitor.visit(this);
    }

    virtual bool check_type(std::type_info const &)
    {
        return false;
    }
};

template<typename TAcceptor, typename ...Args>
struct Visitor : public VisitorBase
{
    std::string name_;
    typedef TAcceptor acceptor_type;
public:

    typedef std::tuple<Args...> args_tuple_type;
    std::tuple<Args...> args_;

    Visitor(Args ... args)
            : args_(std::make_tuple(args...))
    {
    }

    ~Visitor()
    {
    }


    void visit(AcceptorBase &obj)
    {
        if (obj.check_type(typeid(acceptor_type)))
        {
            reinterpret_cast<acceptor_type &>(obj)->template accept(*this);
        }
        else
        {
            THROW_EXCEPTION << "acceptor type mismatch" << std::endl;
        }
    }

    template<typename TFUN>
    inline void execute(TFUN const &f)
    {
        callFunc(f, typename make_index_sequence<sizeof...(Args)>::type());
    }

private:
// Unpack tuple to args...
//\note  http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer


    template<typename TFUN, unsigned int  ...S>
    inline void callFunc(TFUN const &fun, integer_sequence<int, S...>)
    {
        fun(std::get<S>(args_) ...);
    }

};

template<typename T, typename ...Args>
std::shared_ptr<VisitorBase> createVisitor(std::string const &name, Args ...args)
{
    return std::dynamic_pointer_cast<VisitorBase>(std::shared_ptr<Visitor<T, Args...>>(

            new Visitor<T, Args...>(name, std::forward<Args>(args)...)));
}

/** @}*/
} // namespace simpla

#endif /* VISITOR_H_ */
