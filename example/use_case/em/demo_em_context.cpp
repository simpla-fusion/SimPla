/*
 * demo_em_context.cpp
 *
 *  Created on: 2015年1月4日
 *      Author: salmon
 */

#include "../../../core/application/context.h"
namespace simpla
{

class EMContext: public Context
{
public:
	std::vector<Context> split(size_t num_of_intervals)
	{
	}
	;

	void setup(int argc = 0, char const ** argv = nullptr);
	void body();
	void sync();
};

void EMContext::setup(int argc, char const ** argv)
{

}
void EMContext::body()
{

}
void EMContext::sync()
{

}
}  // namespace simpla
