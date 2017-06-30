/**
 * @file MPIAuxFunctions.h
 *
 * @date    2014-7-18  3:42:53
 * @author salmon
 */

#ifndef MPI_AUX_FUNCTIONS_H_
#define MPI_AUX_FUNCTIONS_H_

#include <simpla/SIMPLA_config.h>

#include <stddef.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

//#include <simpla/data/DataSet.h>
#include <simpla/utilities/nTuple.h>

namespace simpla {
namespace parallel {

void bcast_string(std::string *filename_);

// std::tuple<std::shared_ptr<byte_type>, int> update_ghost_unorder(void const *send_buffer,
//                                                                 std::vector<std::tuple<int,  // dest;
//                                                                         int,  // send_tag;
//                                                                         int,  // recv_tag;
//                                                                         int,  // send m_buffer begin;
//                                                                         int   // send m_buffer size;
//                                                                 >> const &info);

// void reduce(void const *send_data, void *recv_data, size_t count, data::DataType const &array_type,
//            std::string const &op_c);
//
// void allreduce(void const *send_data, void *recv_data, size_t count, data::DataType const &array_type,
//               std::string const &op_c);
//
// template <typename T>
// void reduce(T *send_data, T *recv_data, size_t count, std::string const &op_c = "Sum") {
//    reduce(send_data, recv_data, count, data::DataType::create<T>(), op_c);
//}
//
// template <typename T>
// void allreduce(T *send_data, T *recv_data, size_t count, std::string const &op_c = "Sum") {
//    allreduce(send_data, recv_data, count, data::DataType::create<T>(), op_c);
//}
//
// template <typename T>
// T reduce(T send, std::string const &op_c = "Sum") {
//    T recv;
//    reduce(&send, &recv, 1, op_c);
//    return recv;
//}
//
// template <typename T>
// void reduce(T *p_send, std::string const &op_c = "Sum") {
//    T recv;
//    reduce(p_send, &recv, 1, op_c);
//    *p_send = recv;
//}
//
// template <typename T, int DIMS>
// nTuple<T, DIMS> reduce(nTuple<T, DIMS> const &send, std::string const &op_c = "Sum") {
//    nTuple<T, DIMS> recv;
//    reduce(&send[0], &recv[0], DIMS, op_c);
//    return recv;
//}
//
// template <typename T, int DIMS>
// void reduce(nTuple<T, DIMS> *p_send, std::string const &op_c = "Sum") {
//    nTuple<T, DIMS> recv;
//    reduce(&(*p_send)[0], &recv[0], DIMS, op_c);
//    *p_send = recv;
//}
//
// template <typename T>
// T allreduce(T send, std::string const &op_c = "Sum") {
//    T recv;
//    allreduce(&send, &recv, 1, op_c);
//    return recv;
//}
//
// template <typename T>
// void allreduce(T *p_send, std::string const &op_c = "Sum") {
//    T recv;
//    allreduce(p_send, &recv, 1, op_c);
//    *p_send = recv;
//}
//
// template <typename T, int DIMS>
// nTuple<T, DIMS> allreduce(nTuple<T, DIMS> const &send, std::string const &op_c = "Sum") {
//    nTuple<T, DIMS> recv;
//    allreduce(&send[0], &recv[0], DIMS, op_c);
//    return recv;
//}
//
// template <typename T, int DIMS>
// void allreduce(nTuple<T, DIMS> *p_send, std::string const &op_c = "Sum") {
//    nTuple<T, DIMS> recv;
//    allreduce(&(*p_send)[0], &recv[0], DIMS, op_c);
//    *p_send = recv;
//}

}  //{ namespace parallel
}  // namespace simpla
#endif /* MPI_AUX_FUNCTIONS_H_ */
