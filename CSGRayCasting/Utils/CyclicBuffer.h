#pragma once

#include <cstdlib>
#include <cstring>

template<int N>
struct CyclicFloatBuffer
{
	int pos = 0;
	float buffer[N];
	
	CyclicFloatBuffer()
	{
		for (int i = 0; i < N; i++)
			buffer[i] = 0;
	}

	void add(const float& element)
	{
		buffer[pos] = element;
		pos++;
		pos %= N;
	}

	float average()
	{
		float sum = 0;
		for (int i = 0; i < N; i++)
		{
			sum += buffer[i];
		}
		return sum / N;
	}

};