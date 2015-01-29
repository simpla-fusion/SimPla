#include <iostream>

template<typename T,  int I>
class Foo
{
 public:
 void foo();
};
template<  int I>
void Foo<int,I>::foo()
{
	 std::cout<<"this is a int and I = "<<I<<std::endl; 
}
template<   int I>
void Foo<double,I>::foo()
{
	 std::cout<<"this is a double and I = "<<I<<std::endl; 
}

int main(int argc, char const *argv[])
{
	/* code */
	Foo<int,1> fi;
	Foo<double,2> fd;

	fi.foo();
	fd.foo();
	return 0;
}