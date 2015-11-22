/**
 * @file multi_task.cpp
 * @author salmon
 * @date 2015-11-21.
 */

#include <tbb/task.h>
#include <tbb/task_group.h>
#include <iostream>

int main(int argc, char **argv)
{
    tbb::task_group group;

    group.run([&]() { std::cout << "First" << std::endl; });
    group.run([&]() { std::cout << "Second" << std::endl; });
    group.run([&]() { std::cout << "Third" << std::endl; });

    group.wait();


}