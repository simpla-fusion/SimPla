/*
 * ndarray_test.cpp
 *
 *  Created on: 2013-7-20
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "include/defs.h"
#include "data_struct/array.h"

using namespace simpla;

template<typename T>
class TestArray: public testing::Test
{
protected:
	virtual void SetUp()
	{
		size_t _dims[] =
		{ 10, 20, 30 };

		std::vector<size_t>(_dims, _dims + NDIMS).swap(dims);
		num_of_ele = 1;
		for (int i = 0; i < NDIMS; ++i)
		{
			num_of_ele *= dims[i];
		}

	}
public:
	size_t num_of_ele;
	static const int NDIMS = 3;
	std::vector<size_t> dims;
};
typedef testing::Types<double, int, std::complex<double> > MyTypes;

TYPED_TEST_CASE(TestArray, MyTypes);

TYPED_TEST(TestArray,Create){
{
	typename Array::Holder holder = Array::create(TestFixture::dims, typeid(TypeParam));

	holder->clear();

	size_t num= holder->get_num_of_element();

	for(size_t s=0;s<num;++s)
	{
		EXPECT_DOUBLE_EQ(0 ,abs(holder->get_value<TypeParam>(s)));
	}

}
}

TYPED_TEST(TestArray,set_get){
{
	typename Array::Holder holder = Array::create(TestFixture::dims, typeid(TypeParam));

	size_t num= holder->get_num_of_element();

	for(size_t s=0;s<num;++s)
	{
		holder->set_value<TypeParam>(s,static_cast<TypeParam>(s));

	}

	for(size_t s=0;s<num;++s)
	{
		EXPECT_DOUBLE_EQ(0 ,abs(static_cast<TypeParam>(s)-holder->get_value<TypeParam>(s)));
	}
}
}

class TestArrayIO: public testing::TestWithParam<size_t>
{
protected:
	virtual void SetUp()
	{
		size_t _dims[] =
		{ 2, 5, 9 };

		num_of_ele = 1;
		dims.resize(NDIMS);
		for (int i = 0; i < NDIMS; ++i)
		{
			dims[i] = _dims[i] * GetParam();
			num_of_ele *= dims[i];
		}

	}
public:
	size_t num_of_ele;
	static const int NDIMS = 3;
	std::vector<size_t> dims;
};

TEST_P(TestArrayIO, raw_array)
{

	std::vector<double> v(num_of_ele);

	size_t num = num_of_ele;

	for (size_t s = 0; s < num; ++s)
	{
		v[s] = static_cast<double>(s);

	}

	for (size_t s = 0; s < num; ++s)
	{
		EXPECT_DOUBLE_EQ(0, static_cast<double>(s)-v[s]);
	}

}
TEST_P(TestArrayIO, array)
{

	typename Array::Holder holder = Array::create(dims, typeid(double));

	size_t num = num_of_ele;

	for (size_t s = 0; s < num; ++s)
	{
		holder->set_value(s, static_cast<double>(s));

	}

	for (size_t s = 0; s < num; ++s)
	{
		EXPECT_DOUBLE_EQ(0, static_cast<double>(s)-holder->get_value<double>(s))
			;
	}
}

INSTANTIATE_TEST_CASE_P(TestPerformance, TestArrayIO, testing::Values(5, 20, 50));

//TEST_F(ArrayTest, Create)
//{
//
//	dims = 10;
//	ArrayType a(dims);
//	EXPECT_EQ(a.getNumOfDims(), ndims);
//	for (int i = 0; i < ndims; ++i)
//	{
//		EXPECT_EQ(a.getDims()[i], dims[i]);
//	}
//	for (DomainType it = a.getDomain(); !it.isEnd(); ++it)
//	{
//		a[it] = it[0] + it[1] * 10;
//	}
//	ArrayType c(a);
//	int cycle_ = 0;
//	for (DomainType it = a.getDomain(); !it.isEnd(); ++it)
//	{
//		EXPECT_EQ(a[it], it[0]+it[1]*10);
//		++cycle_;
//		EXPECT_EQ(a[it], c[it]);
//	}
//	EXPECT_EQ(cycle_,dims[0]*dims[1])
//		<< "中心(不包括ghost)遍历不完整";
//}
//TEST_F(ArrayTest2DScalar, SubArray)
//{
//	typename ArrayType::IndexType dims;
//	dims = 20;
//	ArrayType a(dims);
//	for (DomainType it = a.getDomain(); !it.isEnd(); ++it)
//	{
//		a[it] = it[0] + it[1] * 10;
//	}
//	cout << a << endl;
//	typename ArrayType::IndexType cycle_ =
//	{ 8, 5 }, start =
//	{ 3, 2 }, stride =
//	{ 2, 3 };
//	ArrayType b(a.subArray(cycle_, start, stride));
//	for (DomainType it = b.getDomain(); !it.isEnd(); ++it)
//	{
//		typename ArrayType::IndexType _J;
//		_J = stride * it + start;
//		EXPECT_EQ(a[_J], b[it]);
//	}
//	b = 10;
//	cout << a << endl << b << endl;
//}
//TEST_F(ArrayTest2DScalar, Arithmetic)
//{
//	typename ArrayType::IndexType dims;
//	dims = 200;
//	ArrayType a(dims), b(dims), c(dims);
//	for (DomainType it = a.getDomain(); !it.isEnd(); ++it)
//	{
//		a[it] = it[0] + it[1] * 10;
//	}
//	Real s = 3.14159;
//	a *= s;
//	b = 1.0;
//	c = 3.0;
//	a = a + b / 2 * c;
//	for (DomainType it = a.getDomain(); !it.isEnd(); ++it)
//	{
//		EXPECT_EQ(a[it], (it[0]+it[1]*10)*s+b[it]/2*c[it]);
//	}
//}
//TEST_F(ArrayTest2DScalar, Ghost)
//{
//    for(DomainType it= b.getDims();!it.isEnd();++it)
//    {
//        a[it]=it[0]+it[1]*10;
//    }
//    IndexType offset  ;
//    offset=dims.raw_;
//
//    a.intersect(a,offset,ARRAY_SYN_GET);
//    offset[0]=0;
//    a.intersect(a,offset,ARRAY_SYN_GET);
//
//    cout<<"=== Here need some autotest Code! ==="<<endl;
//
//    for(ArrayType::iterator it= a.getIterator();!it.isEnd();++it)
//    {
//        a[it]=it[0]+it[1]*10;
//    }
//
//    for(ArrayType::iterator it= a.getFullIterator();!it.isEnd();++it)
//    {
//        if( it[0]== it.begin_[0]){cout<<endl;}
//        //        cout<<it.IDX_<<endl;
//        cout << _s("%2.f ")%(a[it]);
//    }
//    cout<<endl<<"===================================="<<endl;
//
//}
//TEST (ArrayTest1DVec3, synchronize)
//{
//    Tuple<SizeType,ONE> dims={8};
//    Tuple<SizeType,ONE> gw={3};
//    Array<RealVec3,ONE> a(dims,gw);
//    a.self_.fullClear();
//    for(Array<RealVec3,ONE>::iterator _I= a.getIterator();!_I.isEnd();++_I)
//    {
//        a[_I]=_I[0];
//    }
//
//    for(Array<RealVec3,ONE>::iterator _I= a.getFullIterator();!_I.isEnd();++_I)
//    {
//        cout<<  a[_I]<<" ";
//    }
//    dims[0]-=1;
//    a.intersect(a,dims,ARRAY_SYN_ADD_AND_SET);
//    dims =-(dims  );
//    a.intersect(a,dims,ARRAY_SYN_ADD_AND_SET);
//    cout<<"\n======"<<endl;
//    for(Array<RealVec3,ONE>::iterator _I= a.getFullIterator();!_I.isEnd();++_I)
//    {
//        cout<<  a[_I]<<" ";
//    }
//    cout<<endl;
//
//
//}
//
//TEST_F(ArrayTest2DScalar, MultiThreadWrite)
//{
//    Real s=2.0;
//    int MAX_NUM_THREADS=4;
//    omp_set_num_threads(MAX_NUM_THREADS);
//
//    {
//        omp_lock_t lock;
//        omp_init_lock(&lock);
//
//#pragma omp parallel
//        {
//            int num_threads = omp_get_num_threads();
//
//            int thread_num = omp_get_thread_num();
//
//
//
//            MultiThreadWrite<Array<Real,TWO> > A(a,lock);
//
//            Tuple<SizeType,TWO> idx;
//
//            for(ArrayType::iterator it= a.getIterator();!it.isEnd();++it)
//            {
//                A.add(it,s);
//            }
//        }
//        omp_destroy_lock(&lock);
//    }
//
//    for(ArrayType::iterator it= a.getIterator();!it.isEnd();++it)
//    {
//        if(a[it]!= MAX_NUM_THREADS*s){THROW("错误！")};
//        //        EXPECT_EQ(a[it], MAX_NUM_THREADS*s);
//        //NOTE there is some conflict between googletest and OpenMP with Intel C++ compiler
//        //        if( it[0]== it.begin_[0]){cout<<endl;}cout << _s("%2.f ")%(a[it]);
//    }
//
//}
//
//TEST (ArrayTest2DVec3, typeConvert)
//{
//    static const int ndims=2;
//
//    Tuple<SizeType,ndims> dims ={5,5 };
//
//    Array<Vec3,ndims> a(dims );a.fullClear();
//
//    for(Array<Vec3,ndims>::iterator _I= a.getIterator();!_I.isEnd();++_I)
//    {
//        a[_I][0] = _I[0]*10 + _I[1]*100+0 ;
//        a[_I][1] = _I[0]*10 + _I[1]*100+1 ;
//        a[_I][2]= _I[0]*10 + _I[1]*100+2 ;
//    }
//
//    Array<Real,ndims+1> b(a );
//    //    CHECK(_s("rank %d [%d %d]")%b.getRank()%b.getDims()[0]%b.getDims()[1]);
//    for(Array<Real,ndims+1>::iterator _I= b.getIterator();!_I.isEnd();++_I)
//    {
//        EXPECT_EQ(  b[_I], _I[0]*10 + _I[1]*100+_I[2] );
//    }
//
//    Array<Vec3,ndims> c(b );
//    for(Array<Vec3,ndims>::iterator _I= c.getIterator();!_I.isEnd();++_I)
//    {
//        EXPECT_EQ(c[_I][0],_I[0]*10 + _I[1]*100+0);
//        EXPECT_EQ(c[_I][1],_I[0]*10 + _I[1]*100+1);
//        EXPECT_EQ(c[_I][2],_I[0]*10 + _I[1]*100+2);
//    }
//}

