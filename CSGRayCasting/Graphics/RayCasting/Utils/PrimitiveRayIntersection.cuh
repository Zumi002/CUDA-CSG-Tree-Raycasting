#pragma once

#include "Ray.cuh"
#include "../CSGTree/CSGTree.cuh"

//
// ---- headers ----
//

//functions which gets intersection data with given sphere
inline __device__ bool sphereHit(
	const Ray& ray,
	const CudaPrimitivePos& shperePos,
	const Parameters& shpereParams,
	RayHitMinimal& hitInfo, float& tmin
);

//functions which gets details about given intersection data  
inline __device__ void sphereHitDetails(
	const Ray& ray,
	const CudaPrimitivePos& shperePos,
	const Parameters& shpereParams,
	const RayHitMinimal& hitInfo,
	RayHit& detailedHitInfo
);

//same for cylinder
inline __device__ bool cylinderHit(
	const Ray& ray,
	const CudaPrimitivePos& cylinderPos,
	const Parameters& cylinderParams,
	RayHitMinimal& hitInfo, float& tmin
);

inline __device__ void cylinderHitDetails(
	const Ray& ray,
	const CudaPrimitivePos& cylinderPos,
	const Parameters& cylinderParams,
	const RayHitMinimal& hitInfo,
	RayHit& detailedHitInfo
);

//same for cube
inline __device__ bool cubeHit(
	const Ray& ray,
	const CudaPrimitivePos& cubePos,
	const Parameters& cubeParams,
	RayHitMinimal& hitInfo, float& tmin
);

inline __device__ void cubeHitDetails(
	const Ray& ray,
	const CudaPrimitivePos& cubePos,
	const Parameters& cubeParams,
	const RayHitMinimal& hitInfo,
	RayHit& detailedHitInfo
);


//
// ---- code ----
//

inline __device__ bool sphereHit(
	const Ray& ray,
	const CudaPrimitivePos& shperePos,
	const Parameters& shpereParams,
	RayHitMinimal& hitInfo, float& tmin
)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	float3 oc = make_float3(
		ray.origin.x - shperePos.x,
		ray.origin.y - shperePos.y,
		ray.origin.z - shperePos.z
	);

	float b = dot(oc, ray.direction);
	float c = dot(oc, oc) - shpereParams.sphereParameters.radius * shpereParams.sphereParameters.radius;
	float discriminant = b * b - c;

	if (discriminant < 0) return false;

	float temp = (-b - sqrtf(discriminant));
	if (temp <= tmin) {
		temp = (-b + sqrtf(discriminant));
		if (temp <= tmin)
		{
			hitInfo.t = -1;
			hitInfo.hit = CSG::CSGRayHit::Miss;
			hitInfo.primitiveIdx = -1;
			return false;
		}
	}

	hitInfo.t = temp;

	float3 normal = ray.computePosition(temp);

	normal =
		make_float3(
			normal.x - shperePos.x,
			normal.y - shperePos.y,
			normal.z - shperePos.z);

	if (dot(normal, ray.direction) <= 0)
		hitInfo.hit = CSG::CSGRayHit::Enter;
	else
		hitInfo.hit = CSG::CSGRayHit::Exit;

	hitInfo.primitiveType = CSGTree::NodeType::Sphere;

	return true;
}

inline __device__  void sphereHitDetails(
	const Ray& ray,
	const CudaPrimitivePos& shperePos,
	const Parameters& shpereParams,
	const RayHitMinimal& hitInfo,
	RayHit& detailedHitInfo
)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.t = hitInfo.t;
	detailedHitInfo.position = ray.computePosition(detailedHitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;
	detailedHitInfo.normal = normalize(
		make_float3(
			detailedHitInfo.position.x - shperePos.x,
			detailedHitInfo.position.y - shperePos.y,
			detailedHitInfo.position.z - shperePos.z
		)
	);
	if (hitInfo.hit & CSG::CSGRayHit::Flip)
		detailedHitInfo.normal = -detailedHitInfo.normal;
	if (hitInfo.hit & CSG::CSGRayHit::Exit)
		detailedHitInfo.normal = -detailedHitInfo.normal;
}

inline __device__ bool cylinderHit(
	const Ray& ray,
	const CudaPrimitivePos& cylinderPos,
	const Parameters& cylinderParams,
	RayHitMinimal& hitInfo, float& tmin
)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	CylinderParameters params = cylinderParams.cylinderParameters;
	float3 V = make_float3(params.axisX, params.axisY, params.axisZ);
	float3 C = make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (params.height / 2) * V;
	float3 OC = ray.origin - C;

	float aHelp = dot(ray.direction, V);
	float a = fmax(1 - aHelp * aHelp, 0.00001f);

	float cHelp = dot(OC, V);
	float c = dot(OC, OC) - cHelp * cHelp - params.radius * params.radius;

	float b = (dot(ray.direction, OC) - dot(ray.direction, V) * dot(OC, V));
	float discriminant = b * b - a * c;

	if (discriminant < 0)
		return false;


	float t1 = (-b - sqrtf(discriminant)) / (a), t2 = (-b + sqrtf(discriminant)) / (a);
	float m1 = t1 * dot(ray.direction, V) + dot(OC, V), m2 = t2 * dot(ray.direction, V) + dot(OC, V);



	if ((m1 < 0 && m2 < 0) || (m1 > params.height && m2 > params.height))
		return false;

	float temp = t1;
	float m = m1;

	bool skip = false;
	float3 surfNormal;
	int useSurfNormal = 0;
	if (m < 0)
	{
		float den = dot(ray.direction, -V);
		if (fabsf(den) < 0.0001f)
		{
			skip = true;
		}
		else
		{
			temp = dot(-OC, -V) / den;
			surfNormal = -V;
			useSurfNormal = 1;
		}
	}
	if (m > params.height)
	{
		float den = dot(ray.direction, V);
		if (fabsf(den) < 0.0001f)
		{
			skip = true;
		}
		else
		{
			float3 OCmMax = ray.origin - make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (params.height / 2) * V;
			temp = dot(-OCmMax, V) / den;
			surfNormal = V;
			useSurfNormal = 2;
		}
	}
	if (temp <= tmin || skip)
	{
		temp = t2;
		m = m2;
		skip = false;
		useSurfNormal = false;
		if (m < 0)
		{
			float den = dot(ray.direction, -V);
			if (fabsf(den) < 0.0001f)
			{
				skip = true;
			}
			else
			{
				temp = dot(-OC, -V) / den;
				surfNormal = -V;
				useSurfNormal = 1;
			}
		}
		if (m > params.height)
		{
			float den = dot(ray.direction, V);
			if (fabsf(den) < 0.0001f)
			{
				skip = true;
			}
			else
			{
				float3 OCmMax = ray.origin - make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (params.height / 2) * V;
				temp = dot(-OCmMax, V) / den;
				surfNormal = V;
				useSurfNormal = 2;
			}
		}
		if (temp <= tmin || skip)
		{
			return false;
		}
	}



	hitInfo.t = temp;
	float3 normal;
	if (useSurfNormal)
	{
		normal = surfNormal;

	}
	else
	{
		normal = ray.computePosition(temp) - C - m * V;
	}

	if (dot(normal, ray.direction) <= 0)
		hitInfo.hit = CSG::CSGRayHit::Enter;
	else
		hitInfo.hit = CSG::CSGRayHit::Exit;

	if (useSurfNormal == 1)
	{
		hitInfo.hit |= CSG::CSGRayHit::Flag1;
	}
	else if (useSurfNormal == 2)
	{
		hitInfo.hit |= CSG::CSGRayHit::Flag2;
	}
	hitInfo.primitiveType = CSGTree::NodeType::Cylinder;
}

inline __device__ void cylinderHitDetails(
	const Ray& ray,
	const CudaPrimitivePos& cylinderPos,
	const Parameters& cylinderParams,
	const RayHitMinimal& hitInfo,
	RayHit& detailedHitInfo
)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.t = hitInfo.t;
	detailedHitInfo.position = ray.computePosition(detailedHitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;

	CylinderParameters params = cylinderParams.cylinderParameters;
	float3 V = make_float3(params.axisX, params.axisY, params.axisZ);
	float3 C = make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (params.height / 2) * V;
	float3 OC = ray.origin - C;


	if (hitInfo.hit & CSG::CSGRayHit::Flag1)
	{
		float den = dot(ray.direction, -V);
		detailedHitInfo.normal = -V;
	}
	else if (hitInfo.hit & CSG::CSGRayHit::Flag2)
	{
		float den = dot(ray.direction, V);
		detailedHitInfo.normal = V;
	}
	else
	{
		float m = dot(ray.direction, V) * hitInfo.t + dot(OC, V);
		detailedHitInfo.normal = normalize(detailedHitInfo.position - C - m * V);
	}


	if (hitInfo.hit & CSG::CSGRayHit::Flip)
		detailedHitInfo.normal = -detailedHitInfo.normal;
	if (hitInfo.hit & CSG::CSGRayHit::Exit)
		detailedHitInfo.normal = -detailedHitInfo.normal;
}

//with help [https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms]
inline __device__ bool cubeHit(
	const Ray& ray,
	const CudaPrimitivePos& cubePos,
	const Parameters& cubeParams,
	RayHitMinimal& hitInfo, float& tmin
)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;



	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray

	float3 axis = make_float3(1, 1, 1);
	float3 C = make_float3(cubePos.x, cubePos.y, cubePos.z);
	float3 lb = C - cubeParams.cubeParameters.size / 2 * axis;
	float3 rt = C + cubeParams.cubeParameters.size / 2 * axis;
	float t1 = (lb.x - ray.origin.x) / ray.direction.x;
	float t2 = (rt.x - ray.origin.x) / ray.direction.x;
	float t3 = (lb.y - ray.origin.y) / ray.direction.y;
	float t4 = (rt.y - ray.origin.y) / ray.direction.y;
	float t5 = (lb.z - ray.origin.z) / ray.direction.z;
	float t6 = (rt.z - ray.origin.z) / ray.direction.z;

	float tempmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
	float tempmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));


	// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
	if (tempmax < 0)
	{
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tempmin > tempmax)
	{
		return false;
	}
	if (tempmin <= tmin)
	{
		tempmin = tempmax;
		if (tempmin <= tmin)
		{
			return false;
		}
	}

	hitInfo.t = tempmin;
	float3 PC = ray.computePosition(tempmin) - C;
	float bias = 1.00001f;
	float halfSize = cubeParams.cubeParameters.size / 2;
	float3 normal = make_float3((float)(int)(PC.x / halfSize * bias), (float)(int)(PC.y / halfSize * bias), (float)(int)(PC.z / halfSize * bias));


	if (dot(normal, ray.direction) <= 0)
		hitInfo.hit = CSG::CSGRayHit::Enter;
	else
		hitInfo.hit = CSG::CSGRayHit::Exit;

	hitInfo.primitiveType = CSGTree::NodeType::Cube;
	return true;
}

inline __device__ void cubeHitDetails(
	const Ray& ray,
	const CudaPrimitivePos& cubePos,
	const Parameters& cubeParams,
	const RayHitMinimal& hitInfo,
	RayHit& detailedHitInfo
)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.t = hitInfo.t;
	detailedHitInfo.position = ray.computePosition(detailedHitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;

	float3 C = make_float3(cubePos.x, cubePos.y, cubePos.z);


	float3 PC = detailedHitInfo.position - C;
	float bias = 1.00001;
	float halfSize = cubeParams.cubeParameters.size / 2;
	float3 normal = make_float3((float)(int)(PC.x / halfSize * bias), (float)(int)(PC.y / halfSize * bias), (float)(int)(PC.z / halfSize * bias));

	detailedHitInfo.normal = normalize(normal);

	if (hitInfo.hit & CSG::CSGRayHit::Flip)
		detailedHitInfo.normal = -detailedHitInfo.normal;
	if (hitInfo.hit & CSG::CSGRayHit::Exit)
		detailedHitInfo.normal = -detailedHitInfo.normal;
}