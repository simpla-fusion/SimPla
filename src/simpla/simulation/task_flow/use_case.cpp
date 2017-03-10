/**
 * @file use_case.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

//#include "use_case.h"
//
//#include <functional>
//#include <iostream>
//#include <map>
//#include <memory>
//#include <string>
//#include <tuple>
//#include <utility>
//
//#include "../toolbox/utilities/utilities.h"
//#include "../parallel/MPIComm.h"


//
//namespace simpla { namespace use_case
//{
////
////std::string UseCaseList::add(std::string const &name,
////                             std::shared_ptr<UseCase> const &p)
////{
////    base_type::operator[](name) = p;
////
////    return "UseCase" + type_cast<std::string>(base_type::size()) + "_" + name;
////}
////
////std::ostream &UseCaseList::print(std::ostream &os)
////{
////    for (auto const &item : *this)
////    {
////        os << item.first << std::endl;
////    }
////    return os;
////}
//
//void UseCase::run()
//{
////    for (size_t step = 0; step < m_num_of_steps_; ++step)
////    {
////        VERBOSE << "Step [" << step << "/" << m_num_of_steps_ << "]" << std::endl;
////
////
////        GLOBAL_COMM.barrier();
////
////        accept_signal();
////
////        body();
////
////        GLOBAL_COMM.barrier();
////
////        // Check Point
////        if (step % m_check_point_ == 0) { checkpoint(); }
////
////        GLOBAL_COMM.barrier();
////    }
//}
//}}  // namespace simpla
