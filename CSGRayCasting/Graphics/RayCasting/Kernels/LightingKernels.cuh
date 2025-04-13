#pragma once

#include <device_launch_parameters.h>
#include "../Utils/Ray.cuh"
#include "../../RenderManager/Camera/Camera.h"
#include "../CSGTree/CSGTree.cuh"
#include "../Utils/CudaCamera.cuh"

//
// ---- headers ----
//

//kernel called once per pixel, calculates output color based of ray intersecion data and lightnigng
__global__ void LightningKernel(CudaCamera cam, RayHit* hits, CudaPrimitiveColor* primitieveColors, float4* output, float3 lightDir, float width, float height);

//
// ---- code ----
//

__global__ void LightningKernel(CudaCamera cam, RayHit* hits, CudaPrimitiveColor* primitieveColors, float4* output, float3 lightDir, float width, float height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIdx = (y * (int)width + x);

	RayHit hitInfo = hits[pixelIdx];



	if (hitInfo.hit)
	{
		float3 color = make_float3(primitieveColors[hitInfo.primitiveIdx].r,
			primitieveColors[hitInfo.primitiveIdx].g,
			primitieveColors[hitInfo.primitiveIdx].b);

		float3 lightColor = make_float3(1.0f, 1.0f, 1.0f);


		float ka = 0.2f;
		float kd = 0.8f;
		float ks = 0.7f;
		float shininess = 30.0f;

		// Calculate lighting vectors

		lightDir = normalize(lightDir);
		float3 viewDir = normalize(cam.position - hitInfo.position);
		float3 reflectDir = reflect(-lightDir, hitInfo.normal);

		// Ambient component
		float3 ambient = ka * lightColor;

		// Diffuse component
		float diff = fmax(dot(hitInfo.normal, lightDir), 0.0f);
		float3 diffuse = kd * diff * lightColor;

		// Specular component
		float spec = pow(fmax(dot(viewDir, reflectDir), 0.0f), shininess);
		float3 specular = ks * spec * lightColor;

		// Combine all components
		float3 finalColor = make_float3(
			color.x * (ambient.x + diffuse.x + specular.x),
			color.y * (ambient.y + diffuse.y + specular.y),
			color.z * (ambient.z + diffuse.z + specular.z)
		);

		// Clamp colors to [0,1]
		finalColor.x = fmin(fmax(finalColor.x, 0.0f), 1.0f);
		finalColor.y = fmin(fmax(finalColor.y, 0.0f), 1.0f);
		finalColor.z = fmin(fmax(finalColor.z, 0.0f), 1.0f);

		output[pixelIdx] = make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f);
	}
	else
	{
		output[pixelIdx] = make_float4(0.08f, 0.08f, 0.11f, 1);
	}
}