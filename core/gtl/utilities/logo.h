/**
 * @file simpla_lib.cpp.h
 * @author salmon
 * @date 2015-10-21.
 */

#ifndef SIMPLA_SIMPLA_LIB_CPP_H
#define SIMPLA_SIMPLA_LIB_CPP_H
#include <string>

namespace simpla
{

std::string ShowLogo();
std::string ShowVersion();
std::string ShowCopyRight();

void TheStart(int flag = 1);
void TheEnd(int flag = 1);

}
#endif //SIMPLA_SIMPLA_LIB_CPP_H
