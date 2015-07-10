//
// Created by salmon on 7/10/15.
//

#ifndef SIMPLA_SIGNAL_H
#define SIMPLA_SIGNAL_H

#include <boost/signals2/signal.hpp>

namespace simpla {

template<typename ...T> using signal= boost::signals2::signal<T...>;

}// namespace simpla
#endif //SIMPLA_SIGNAL_H
