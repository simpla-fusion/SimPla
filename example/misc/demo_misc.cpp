/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */

#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"

using namespace simpla;

struct id_tuple
{
	unsigned long i :20;
	unsigned long j :20;
	unsigned long k :20;
	int n :4;

	template<typename T>
	operator T() const
	{
		return *reinterpret_cast<std::int64_t const *>(this);
	}
};
int main(int argc, char **argv)
{
	SHOW(sizeof(long));
	SHOW(sizeof(std::int32_t));
	SHOW(sizeof(std::int64_t));

	SHOW(sizeof(id_tuple));
	id_tuple t =
	{ 1, 0xF2, 0xEFFFFL, 4 };

	std::cout << std::hex << static_cast<std::int64_t>(t) << std::endl;

}

