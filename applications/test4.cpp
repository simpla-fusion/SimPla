#include <iostream>
#include <string>

#include <memory>
struct C
{
	C()
	{
		std::cout << "Construct" << std::endl;
	}
	~C()
	{
		std::cout << "Destruct" << std::endl;
	}
};
int main(void)
{
	std::shared_ptr<void> p(
			std::static_pointer_cast<void>(std::shared_ptr<C>(new C)));

	std::cout << "lalala" << std::endl;

	p = nullptr;

}
