#pragma once

#include <cuda_runtime.h>

template <typename T, int N>
class CudaStack 
{
	int count = 0;
	public:
	T stack[N];

	inline __device__ void push(const T& element)
	{
		if (size() >= N)
		{
			int stackFull = 0;
//			assert(stackFull);
			return;
		}
		stack[count] = element;
		count++;
	}

	inline __device__ T pop()
	{
		if (empty())
		{
			int stackEmpty = 0;
	//		assert(stackEmpty);
			return stack[count];
		}
		count--;
		return stack[count];
	}

	inline __device__ bool empty()
	{
		return count == 0;
	}

	inline __device__ int size()
	{
		return count;
	}
};