#pragma once

#include <cuda_runtime.h>

template <typename T, int N>
class CudaStack 
{
	int count = 0;
	public:
	T stack[N];

	__device__ void push(const T& element)
	{
		if (size() == N)
		{
			int stackFull = 0;
			assert(stackFull);
		}
		stack[count] = element;
		count++;
	}

	__device__ T pop()
	{
		if (empty())
		{
			int stackEmpty = 0;
			assert(stackEmpty);
		}
		count--;
		return stack[count];
	}

	__device__ bool empty()
	{
		return count == 0;
	}

	__device__ int size()
	{
		return count;
	}
};