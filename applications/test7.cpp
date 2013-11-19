#include "pugixml.hpp"
#include "exprtk.hpp"
#include <iostream>

template<typename T>
struct Function
{

	T x_, y_, z_;
	exprtk::symbol_table<T> symbol_table;
	exprtk::expression<T> expression;
	std::map<std::string, T> constants_;

	Function()
	{
		symbol_table.add_variable("x", x_);
		symbol_table.add_variable("y", y_);
		symbol_table.add_variable("z", z_);
		symbol_table.add_constants();
	}
	void AddConstant(std::string const &name, T const & v)
	{
		constants_[name] = v;
		symbol_table.add_constant(name, constants_[name]);
	}

	void Register(std::string const &expression_string)
	{
		expression.register_symbol_table(symbol_table);
		exprtk::parser<T>().compile(expression_string, expression);
	}

	template<typename TX, typename TY, typename TZ>
	inline T operator()(TX const & px, TY const & py, TZ const & pz)
	{
		x_ = px;
		y_ = py;
		z_ = pz;
		return expression.value();
	}

};

int main(int argc, char** argv)
{
	pugi::xml_document doc;
	doc.load_file(argv[1]);
	doc.print(std::cout);

	double n0 =

			doc.select_single_node(
					"/Xdmf/Domain/Grid/Information[@Name='GlobalVaraible']/Parameter[@Name='n0']").node().attribute(
					"Value").as_double();

	std::cout << n0 << std::endl;

	std::string expr_str =
			doc.select_single_node(
					"/Xdmf/Domain/Grid/Attribute[@Name='B0']/DataItem").node().text().get();

	std::cout << expr_str << std::endl;

	Function<double> fun;
	fun.AddConstant("n0", n0);

	fun.Register(expr_str);
	for (double x = 0.1; x < 2.0; x += 0.1)
	{
		std::cout << fun(x, 0, 0) << std::endl;

	}

//	for (auto & p : nodes)
//	{
//		p.node().print(std::cout);
//	}

}

// vim:et
