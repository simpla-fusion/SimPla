/**
 * @file parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <string>
namespace simpla {
namespace parallel {
void init(int argc, char **argv);
void close();
std::string help_message();
}  //{ namespace parallel
}  // namespace simpla

#endif /* PARALLEL_H_ */
