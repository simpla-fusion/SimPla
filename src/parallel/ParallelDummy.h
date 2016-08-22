/**
 * @file ParallelDummy.h
 *
 *  Created on: 2014-11-6
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_PARALLEL_DUMMY_H_
#define CORE_PARALLEL_PARALLEL_DUMMY_H_

#include <type_traits>
#include <functional>

namespace simpla { namespace serial
{

namespace tags
{
struct split { };

struct proportional_split
{
    size_t left() { return m_left_; }

    size_t right() { return m_right_; }

private:
    size_t m_left_, m_right_;
};
}


template<typename ...> struct concurrent_unordered_map;
template<typename ...> struct concurrent_unordered_set;
template<typename ...> struct concurrent_hash_map;

template<typename Range, typename OP>
void parallel_do(Range const &range, OP const &op)
{
    if (!range.empty()) op();
}

/**
 *
 * @param entity_id_range RangeHolder Concept
 * @param op std::function<void(RangeHolder)>
 */
template<typename Range, typename Body>
void parallel_for(Range const &range, Body const &op)
{
    op(range);
}

///**
// *
// * @_fdtd_param entity_id_range RangeHolder Concept
// * @_fdtd_param op std::function<void(*RangeHolder::iterator)>
// */
//template<typename RangeHolder, typename Body>
//void parallel_foreach(RangeHolder const &entity_id_range, Body const &body)
//{
//	for (auto &&i : entity_id_range)
//	{
//		body(i);
//	}
//}
//
//template<typename Body>
//void parallel_foreach(size_t b, size_t e, Body const &body)
//{
//	for (size_t i = b; i < e; ++i)
//	{
//		body(i);
//	}
//}

/**
 *
 * @param entity_id_range  RangeHolder Concept
 * @param op     std::function<T(RangeHolder)>
 * @param reduce std::function<T(T,T)>
 * @return T
 */
template<typename Value, typename Range, typename OP, typename Reduction>
auto parallel_reduce(const Range &range, OP const &op,
                     const Reduction &reduce) ->
typename std::result_of<OP(Range const &)>::type
{
    return op(range);
}

/**
 *
 * @param entity_id_range  RangeHolder Concept
 * @param op     std::function<T(RangeHolder)>
 * @return T
 */
template<typename Value, typename Range, typename OP>
auto parallel_reduce(const Range &range, OP const &op) ->
typename std::result_of<OP(Range const &)>::type
{

    typedef typename std::result_of<OP(Range const &)>::type res_type;

    return parallel_reduce(range, op, std::plus<res_type>());
}
}} // namespace simpla { namespace parallel


#endif /* CORE_PARALLEL_PARALLEL_DUMMY_H_ */
