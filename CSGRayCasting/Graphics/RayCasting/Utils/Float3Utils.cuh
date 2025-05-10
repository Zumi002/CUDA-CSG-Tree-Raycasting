#pragma once

#include <cuda_runtime.h>
#include <cmath>

inline __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


inline __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator*(float s, float3 a)
{
	return make_float3(s * a.x, s * a.y, s * a.z);
}


inline __device__ float3 operator-(float3 a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ float3 normalize(float3 a)
{
	float invLen = 1.0f / sqrt(dot(a, a));
	return invLen * a;
}

inline __device__ float3 reflect(float3 a, float3 b)
{
	float3 n = normalize(b);
	float3 ret = a - 2.0f * (dot(a, n)) * n;
	return ret;
}

inline __device__ float length(float3 a)
{
	return sqrtf(dot(a, a));
}

inline __device__ float3 abs(float3 a)
{
    return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}

inline __device__ float3 max(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
