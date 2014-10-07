/**
 * \file map_list.cpp
 *
 * \date    2014年8月29日  下午1:34:05 
 * \author salmon
 */

#include <scoped_allocator>
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <limits>
#include <memory>
template<typename T>
class mt_alloc
{
public:
public:
	// type definitions
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef T value_type;

	std::shared_ptr<std::allocator<T>> base_alloc_;

	// constructors
	// - nothing to do because the allocator has no state
	mt_alloc()
			: base_alloc_(std::make_shared<std::allocator<T>>(0))
	{
		std::cout << "Construct" << std::endl;
	}
	template<typename U>
	mt_alloc(const mt_alloc<U>& other)
			: count_(other.count_)
	{
		++(*count_);
		std::cout << "Copy Construct " << *count_ << std::endl;
	}

	~MyAlloc() throw ()
	{
	}

	// allocate but don't initialize num elements of type T
	T* allocate(std::size_t num, const void* hint = 0)
	{
		// allocate memory with global new
		return static_cast<T*>(::operator new(num * sizeof(T)));
	}

	// deallocate storage p of deleted elements
	void deallocate(T* p, std::size_t num)
	{
		// deallocate memory with global delete
		::operator delete(p);
	}

	// return address of values
	T* address(T& value) const
	{
		return &value;
	}
	const T* address(const T& value) const
	{
		return &value;
	}

	// return maximum number of elements that can be allocated
	std::size_t max_size() const throw ()
	{
		return std::numeric_limits<std::size_t>::max() / sizeof(T);
	}

	// initialize elements of allocated storage p with value value
	void construct(T* p, const T& value)
	{
		// initialize memory with placement new
		::new ((void*) p) T(value);
	}

	// destroy elements of initialized storage p
	void destroy(T* p)
	{
		// destroy objects by calling their destructor
		p->~T();
	}

	// rebind allocator to type U
	template<typename U>
	struct rebind
	{
		typedef MyAlloc<U> other;
	};
};
typedef std::scoped_allocator_adaptor<
        std::allocator<std::pair<const size_t, std::list<double, std::allocator<double>> > >, std::allocator<double>> alloc_type;
typedef std::map<size_t, std::list<double>, std::less<size_t>, alloc_type> pool;
int main(int argc, char **argv)
{
//	std::vector<double, MyAlloc<double>> v;
//	v.push_back(1.0);
//	std::list<double, MyAlloc<double>> l;
//	l.push_back(1.0);
	pool p;
	p[0].push_back(1.0);
	p[1].push_back(2.0);
	p[0].push_back(1.0);
	p[1].push_back(2.0);
	p[0].push_back(1.0);
	p[3].push_back(2.0);
}
