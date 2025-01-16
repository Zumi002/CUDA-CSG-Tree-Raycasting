#pragma once

#include <math.h>
#include <cmath>
#include <cuda_runtime.h>


struct DirectionalLight
{
	float polar = -60.f * 3.14159f / 180.f,
		azimuth = -45.f * 3.14159f / 180.f;

	float3 getLightDir()
	{
		return make_float3(sin(polar) * cos(azimuth),
						   cos(polar),
						   sin(polar) * sin(azimuth));
	}
};