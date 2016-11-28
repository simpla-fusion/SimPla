/**
 * @file parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

/**
 *  @addtogroup  parallel parallel
 *  @{
 *  	@addtogroup  MPI MPI Communicaion
 *  	@addtogroup  MULTICORE Multi-thread/src and many-src support
 *  @}
 */

#include "MPIComm.h"


//#ifdef TBB_FOUND

#   include "ParallelTbb.h"

//#endif

#ifdef OPENMP_FOUND

#   include "ParallelOpenMP.h"

#else

#   include "ParallelDummy.h"

#endif

#include "DistributedObject.h"

namespace simpla { namespace parallel
{
void init(int argc, char **argv);

void close();

std::string help_message();

namespace detail
{

template<typename T, typename Body>
struct foreach_dispatch
{

    HAS_CONST_MEMBER_FUNCTION(parallel_foreach)

    HAS_CONST_MEMBER_FUNCTION(serial_foreach)

    HAS_CONST_MEMBER_FUNCTION(foreach)

    HAS_CONST_MEMBER_FUNCTION(begin)

    HAS_CONST_MEMBER_FUNCTION(end)

    static constexpr size_t value =
            (has_const_member_function_parallel_foreach<T, Body const &>::value ? (0x10UL) : (0UL))  // 0b 0001 0000
            | (has_const_member_function_serial_foreach<T, Body const &>::value ? (0x08UL) : (0UL))  // 0b 0000 1000
            | (has_const_member_function_foreach<T, Body const &>::value/*   */ ? (0x04UL) : (0UL))  // 0b 0000 0100
            | (has_const_member_function_begin<T>::value/*                   */ ? (0x02UL) : (0UL))  // 0b 0000 0010
            | (has_const_member_function_end<T>::value/*                     */ ? (0x01UL) : (0UL))  // 0b 0000 0001


    ;
};
}

/**
 * has parallel_foreach
 */
template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x10UL) == 0x10UL>::type * = nullptr>
void parallel_foreach(TRange const &r, Body const &body)
{
    r.parallel_foreach(body);
}

template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x18UL) == 0x08UL>::type * = nullptr>
void parallel_foreach(TRange const &r, Body const &body)
{
    parallel_for(r, [&](TRange const &r1) { r1.serial_foreach(body); });
}

template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x1CUL) == 0x04UL>::type * = nullptr>
void parallel_foreach(TRange const &r, Body const &body)
{
    parallel_for(r, [&](TRange const &r1) { r1.foreach(body); });
}

template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x1FUL) == 0x03UL>::type * = nullptr>
void parallel_foreach(TRange const &r, Body const &body)
{
    parallel_for(r, [&](TRange const &r1) { for (auto const &s: r1) { body(s); }});
}
//
//template<typename TRange, typename Body,
//        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::entity & 0x1FUL) == 0x10UL>::type * = nullptr>
//void serial_foreach(TRange const &r, Body const &body)
//{
//    UNIMPLEMENTED;
//}

template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x0CUL) == 0x08UL>::type * = nullptr>
void serial_foreach(TRange const &r, Body const &body)
{
    r.serial_foreach(body);
}

template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x0CUL) == 0x04UL>::type * = nullptr>
void serial_foreach(TRange const &r, Body const &body)
{
    r.foreach(body);
}

template<typename TRange, typename Body,
        typename std::enable_if<(detail::foreach_dispatch<TRange, Body>::value & 0x1FUL) == 0x03UL>::type * = nullptr>
void serial_foreach(TRange const &r, Body const &body)
{
    for (auto const &s: r) { body(s); }
}

template<typename TRange, typename Body>
void foreach(TRange const &r, Body const &body)
{
#ifndef NDEBUG
    serial_foreach(r, body);
#else
    parallel_foreach(r, body);
#endif
}
}}// namespace simpla { namespace parallel

#endif /* PARALLEL_H_ */
