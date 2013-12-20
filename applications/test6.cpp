#include "../src/utilities/log.h"

int main(int argc, char **argv)
{
	CHECK(5 % 1);
	CHECK(1 % 1);
	CHECK(0 % 1);
	CHECK(2 % 1);

	CHECK(5 % 101);
	CHECK(100 % 101);
	CHECK(101 % 101);
	CHECK(102 % 101);
}

