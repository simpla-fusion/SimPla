/*
 * logo.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_LOGO_H_
#define CORE_APPLICATION_LOGO_H_

#include <string>

namespace simpla
{

std::string ShowLogo();
std::string ShowVersion();
std::string ShowCopyRight();

void TheStart(int flag = 1);
void TheEnd(int flag = 1);

}

#endif /* CORE_APPLICATION_LOGO_H_ */
