#pragma once

#include "Ray.cuh"
#include "../CSGTree/CSGTree.cuh"

//
// ---- headers ----
//

//decicdes which intersection method is called for given primitive node
inline __device__ void hitPrimitive(
	const Ray& ray,
	const CudaPrimitivePos* __restrict__ primitvePos,
	const Parameters* __restrict__ primitiveParameters,
	const CSGNode& node,
	RayHitMinimal& hitInfo, float tmin
);

//functions which gets intersection data with given sphere
 __device__ bool sphereHit(
	Ray ray,
	CudaPrimitivePos shperePos,
	Parameters shpereParams,
	RayHitMinimal& hitInfo, float tmin
);

//functions which gets details about given intersection data  
 __device__ void sphereHitDetails(
	Ray ray,
	CudaPrimitivePos shperePos,
	Parameters shpereParams,
	RayHitMinimal hitInfo,
	RayHit& detailedHitInfo
);

//same for cylinder
 __device__ bool cylinderHit(
	Ray ray,
	CudaPrimitivePos cylinderPos,
	Parameters cylinderParams,
	RayHitMinimal& hitInfo, float tmin
);

 __device__ void cylinderHitDetails(
	Ray ray,
	CudaPrimitivePos cylinderPos,
	Parameters cylinderParams,
	RayHitMinimal hitInfo,
	RayHit& detailedHitInfo
);

//same for cube
 __device__ bool cubeHit(
	Ray ray,
	CudaPrimitivePos cubePos,
	Parameters cubeParams,
	RayHitMinimal& hitInfo, float tmin
);

 __device__ void cubeHitDetails(
	Ray ray,
	CudaPrimitivePos cubePos,
	Parameters cubeParams,
	RayHitMinimal hitInfo,
	RayHit& detailedHitInfo
);

//
// ---- code ----
//

 inline __device__ void hitPrimitive(
	 const Ray& ray,
	 const CudaPrimitivePos* __restrict__ primitvePos,
	 const Parameters* __restrict__ primitiveParameters,
	 const CSGNode& node,
	 RayHitMinimal& hitInfo,
	 float tmin
 )
 {
	 hitInfo.primitiveIdx = node.primitiveIdx;
	 if (node.type == CSGTree::NodeType::Sphere)
	 {
		 sphereHit(ray, primitvePos[node.primitiveIdx], primitiveParameters[node.primitiveIdx], hitInfo, tmin);
	 }
	 else if (node.type == CSGTree::NodeType::Cylinder)
	 {
		 cylinderHit(ray, primitvePos[node.primitiveIdx], primitiveParameters[node.primitiveIdx], hitInfo, tmin);
	 }
	 else if (node.type == CSGTree::NodeType::Cube)
	 {
		 cubeHit(ray, primitvePos[node.primitiveIdx], primitiveParameters[node.primitiveIdx], hitInfo, tmin);
	 }
	 else
	 {
		 hitInfo.hit = CSG::CSGRayHit::Miss;
	 }
 }

 __device__ bool sphereHit(
	Ray ray,
	CudaPrimitivePos shperePos,
	Parameters shpereParams,
	RayHitMinimal& hitInfo, float tmin
)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	float3 oc = make_float3(
		ray.origin.x - shperePos.x,
		ray.origin.y - shperePos.y,
		ray.origin.z - shperePos.z
	);

	float b = dot(oc, ray.direction);
	float discriminant = b * b - dot(oc, oc) + shpereParams.sphereParameters.radius * shpereParams.sphereParameters.radius;

	if (discriminant < 0) return false;

	b = (-b - sqrtf(discriminant));
	if (b <= tmin) {
		b += 2 * sqrtf(discriminant);
		if (b <= tmin)
		{
			hitInfo.hit = CSG::CSGRayHit::Miss;
			return false;
		}
	}

	hitInfo.t = b;

	float3 normal = ray.computePosition(b);

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

 __device__  void sphereHitDetails(
	Ray ray,
	CudaPrimitivePos shperePos,
	Parameters shpereParams,
	RayHitMinimal hitInfo,
	RayHit& detailedHitInfo
)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.position = ray.computePosition(hitInfo.t);
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

 __device__ bool cylinderHit(
	Ray ray,
	CudaPrimitivePos cylinderPos,
	Parameters cylinderParams,
	RayHitMinimal& hitInfo, float tmin
)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	float3 V = make_float3(cylinderParams.cylinderParameters.axisX, cylinderParams.cylinderParameters.axisY, cylinderParams.cylinderParameters.axisZ);
	float3 C = make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (cylinderParams.cylinderParameters.height / 2) * V;
	float3 OC = ray.origin - C;

	float a= dot(ray.direction, V);
	a = fmax(1 - a * a, 0.00001f);

	float c = dot(OC, V);
	c = dot(OC, OC) - c * c - cylinderParams.cylinderParameters.radius * cylinderParams.cylinderParameters.radius;

	float b = (dot(ray.direction, OC) - dot(ray.direction, V) * dot(OC, V));
	float discriminant = b * b - a * c;

	if (discriminant < 0)
		return false;


	float t1 = (-b - sqrtf(discriminant)) / (a), t2 = (-b + sqrtf(discriminant)) / (a);
	float m1 = t1 * dot(ray.direction, V) + dot(OC, V), m2 = t2 * dot(ray.direction, V) + dot(OC, V);



	if ((m1 < 0 && m2 < 0) || (m1 > cylinderParams.cylinderParameters.height && m2 > cylinderParams.cylinderParameters.height))
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
	if (m > cylinderParams.cylinderParameters.height)
	{
		float den = dot(ray.direction, V);
		if (fabsf(den) < 0.0001f)
		{
			skip = true;
		}
		else
		{
			float3 OCmMax = ray.origin - make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (cylinderParams.cylinderParameters.height / 2) * V;
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
		if (m > cylinderParams.cylinderParameters.height)
		{
			float den = dot(ray.direction, V);
			if (fabsf(den) < 0.0001f)
			{
				skip = true;
			}
			else
			{
				float3 OCmMax = ray.origin - make_float3(cylinderPos.x, cylinderPos.y, cylinderPos.z) - (cylinderParams.cylinderParameters.height / 2) * V;
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

 __device__ void cylinderHitDetails(
	Ray ray,
	CudaPrimitivePos cylinderPos,
	Parameters cylinderParams,
	RayHitMinimal hitInfo,
	RayHit& detailedHitInfo
)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.position = ray.computePosition(hitInfo.t);
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
 __device__ bool cubeHit(
	Ray ray,
	CudaPrimitivePos cubePos,
	Parameters cubeParams,
	RayHitMinimal& hitInfo, float tmin
)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;



	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray

	float3 move = 0.5f * make_float3(cubeParams.cubeParameters.sizeX, cubeParams.cubeParameters.sizeY, cubeParams.cubeParameters.sizeZ);
	float3 C = make_float3(cubePos.x, cubePos.y, cubePos.z);
	float3 lb = C - move;
	float3 rt = C + move;
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
	float bias = 1.0001f;
	float3 normal = make_float3((float)(int)(PC.x / move.x * bias), (float)(int)(PC.y / move.y * bias), (float)(int)(PC.z / move.z * bias));


	if (dot(normal, ray.direction) <= 0)
		hitInfo.hit = CSG::CSGRayHit::Enter;
	else
		hitInfo.hit = CSG::CSGRayHit::Exit;

	hitInfo.primitiveType = CSGTree::NodeType::Cube;
	return true;
}

 __device__ void cubeHitDetails(
	Ray ray,
	CudaPrimitivePos cubePos,
	Parameters cubeParams,
	RayHitMinimal hitInfo,
	RayHit& detailedHitInfo
)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.position = ray.computePosition(hitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;

	float3 C = make_float3(cubePos.x, cubePos.y, cubePos.z);


	float3 PC = detailedHitInfo.position - C;
	float bias = 1.0001f;
	float3 normal = make_float3((float)(int)(PC.x / (cubeParams.cubeParameters.sizeX / 2) * bias), (float)(int)(PC.y / (cubeParams.cubeParameters.sizeY / 2) * bias), (float)(int)(PC.z / (cubeParams.cubeParameters.sizeZ / 2) * bias));

	detailedHitInfo.normal = normalize(normal);

	if (hitInfo.hit & CSG::CSGRayHit::Flip)
		detailedHitInfo.normal = -detailedHitInfo.normal;
	if (hitInfo.hit & CSG::CSGRayHit::Exit)
		detailedHitInfo.normal = -detailedHitInfo.normal;
}