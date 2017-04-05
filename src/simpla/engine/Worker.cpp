//
// Created by salmon on 17-4-5.
//

#include "Worker.h"
namespace simpla {
namespace engine {
struct Worker::pimpl_s {};
Worker::Worker() : m_pimpl_(new pimpl_s) {}
Worker::~Worker() {}

}  // namespace engine{

}  // namespace simpla{