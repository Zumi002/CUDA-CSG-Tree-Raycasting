#pragma once

#include <device_launch_parameters.h>
#include "../Utils/CSGUtils.cuh"
#include "../Utils/Ray.cuh"
#include "../Utils/PrimitiveRayIntersection.cuh"
#include "../Utils/CudaCamera.cuh"

#define NOT_INTERSECTED FLT_MAX
#define DEBUG_PIXEL_X 300
#define DEBUG_PIXEL_Y 300
#define CLASSICSHAREDPRIMITIVES 196
//#define CLASSICRAYCASTING_DEBUG

//
// ---- headers ----
//

__global__ void CalculateInterscetion(
    int width, int height,
    int shape_count,
	CSGNode* nodes,
	CudaPrimitivePos* positions,
	Parameters* primitiveParameters,
    int* parts,
    CudaCamera cam, 
    RayHit* hits
);

__global__ void CalculateInterscetionShared(
	int width, int height,
	int shape_count,
	CSGNode* nodes,
	CudaPrimitivePos* positions,
	Parameters* primitiveParameters,
	int* parts,
	CudaCamera cam,
	RayHit* hits
);

inline __device__ bool IntersectionPointSphere(
    float3 spherePosition,
    float radius,
    float3 rayOrigin,
    float3 rayDirection,
    float& t1, float& t2
);

inline __device__ bool IntersectionPointCube(
	const CudaPrimitivePos& cubePosition,
	const Parameters& cubeParams,
    float3 rayOrigin,
    float3 rayDirection,
    float& t1, float& t2
);

inline __device__ bool IntersectionPointCylinder(
	const CudaPrimitivePos& cylinderPosition,
	const Parameters& cylinderParams,
    float3 rayOrigin,
    float3 rayDirection,
    float& t1, float& t2
);

inline __device__ void AddIntervals(
    float* sphereIntersections,
    float* tempArray,
    int p1, int p2,
    int k1, int k2,
    bool print
);

inline __device__ void SubstractIntervals(
    float* sphereIntersections,
    float* tempArray,
    int p1, int p2,
    int k1, int k2,
    bool print
);

inline __device__ void CommonPartIntervals(
    float* sphereIntersections,
    float* tempArray,
    int p1, int p2,
    int k1, int k2,
    bool print
);

inline __device__ float4 NormalizeVector4(float4 vector);

inline __device__ float3 NormalizeVector3(float3 vector);

inline __device__ float3 cross(float3 a, float3 b);

inline __device__ float dot3(float3 a, float3 b);

inline __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix);

//
// ---- code ----
//

inline __device__ bool IntersectionPointCube(
	const CudaPrimitivePos& cubePosition,
	const Parameters& cubeParams,
	float3 rayOrigin,
	float3 rayDirection,
	float& t1, float& t2
)
{
	float3 move = 0.5f * make_float3(cubeParams.cubeParameters.sizeX, cubeParams.cubeParameters.sizeY, cubeParams.cubeParameters.sizeZ);
	float3 C = make_float3(cubePosition.x, cubePosition.y, cubePosition.z);
	float3 l = C - move;
	float3 h = C + move;
	float3 o = rayOrigin;
	float3 r = rayDirection;

	float t_close;
	float t_far;

	float tx_low = (l.x - o.x) / r.x;
	float tx_high = (h.x - o.x) / r.x;

	float ty_low = (l.y - o.y) / r.y;
	float ty_high = (h.y - o.y) / r.y;

	float tz_low = (l.z - o.z) / r.z;
	float tz_high = (h.z - o.z) / r.z;

	float tx_close = tx_low < tx_high ? tx_low : tx_high;
	float tx_far = tx_low > tx_high ? tx_low : tx_high;

	float ty_close = ty_low < ty_high ? ty_low : ty_high;
	float ty_far = ty_low > ty_high ? ty_low : ty_high;

	float tz_close = tz_low < tz_high ? tz_low : tz_high;
	float tz_far = tz_low > tz_high ? tz_low : tz_high;

	t_close = tx_close > ty_close ? (tx_close > tz_close ? tx_close : tz_close) : (ty_close > tz_close ? ty_close : tz_close);
	t_far = tx_far < ty_far ? (tx_far < tz_far ? tx_far : tz_far) : (ty_far < tz_far ? ty_far : tz_far);

	t1 = t_close;
	t2 = t_far;

	return t_close < t_far;
}

inline __device__ bool IntersectionPointCylinder(
	const CudaPrimitivePos& cylinderPosition,
	const Parameters& cylinderParams,
	float3 rayOrigin,
	float3 rayDirection,
	float& t1, float& t2
)
{
	float3 axis = make_float3(cylinderParams.cylinderParameters.axisX, cylinderParams.cylinderParameters.axisY, cylinderParams.cylinderParameters.axisZ);
	float3 b = make_float3(cylinderPosition.x - rayOrigin.x, cylinderPosition.y - rayOrigin.y, cylinderPosition.z - rayOrigin.z) - (cylinderParams.cylinderParameters.height / 2) * axis;
	float3 a = NormalizeVector3(axis);
	float r = cylinderParams.cylinderParameters.radius;
	float h = cylinderParams.cylinderParameters.height;
	float3 n = rayDirection;

	float d1 = NOT_INTERSECTED; // in line intersection with cylinder
	float d2 = NOT_INTERSECTED; // out line intersection with cylinder
	float d3 = NOT_INTERSECTED; // first cap with line intersection 
	float d4 = NOT_INTERSECTED; // second cap with line intersection 

	float pierw = dot3(cross(n, a), cross(n, a)) * r * r - dot3(a, a) * dot3(b, cross(n, a)) * dot3(b, cross(n, a));
	if (pierw >= 0)
	{
		d1 = (dot3(cross(n, a), cross(b, a))
			- sqrt(pierw))
			/ dot3(cross(n, a), cross(n, a));
		d2 = (dot3(cross(n, a), cross(b, a))
			+ sqrt(pierw))
			/ dot3(cross(n, a), cross(n, a));

		float t11 = dot3(a, make_float3(n.x * d1 - b.x, n.y * d1 - b.y, n.z * d1 - b.z));
		float t22 = dot3(a, make_float3(n.x * d2 - b.x, n.y * d2 - b.y, n.z * d2 - b.z));

		if (!(t11 >= 0 && t11 <= h)) d1 = NOT_INTERSECTED;
		if (!(t22 >= 0 && t22 <= h)) d2 = NOT_INTERSECTED;
	}

	float3 c1 = b;
	float3 c2 = make_float3(b.x + a.x * h, b.y + a.y * h, b.z + a.z * h);

	d3 = dot3(a, c2) / dot3(a, n);
	d4 = dot3(a, c1) / dot3(a, n);

	if (dot3(make_float3(n.x * d3 - c2.x, n.y * d3 - c2.y, n.z * d3 - c2.z), make_float3(n.x * d3 - c2.x, n.y * d3 - c2.y, n.z * d3 - c2.z)) > r * r)
		d3 = NOT_INTERSECTED;
	if (dot3(make_float3(n.x * d4 - c1.x, n.y * d4 - c1.y, n.z * d4 - c1.z), make_float3(n.x * d4 - c1.x, n.y * d4 - c1.y, n.z * d4 - c1.z)) > r * r)
		d4 = NOT_INTERSECTED;

	t1 = NOT_INTERSECTED;
	t2 = NOT_INTERSECTED;
	if (d1 != NOT_INTERSECTED)
	{
		t1 = d1;
	}
	if (d3 != NOT_INTERSECTED && d3 < t1)
	{
		t1 = d3;
	}
	if (d4 != NOT_INTERSECTED && d4 < t1)
	{
		t1 = d4;
	}

	// finding smallest t2
	if (d2 != NOT_INTERSECTED)
	{
		t2 = d2;
	}
	if (d3 != NOT_INTERSECTED && d3 < t2 && d3 != t1)
	{
		t2 = d3;
	}
	if (d4 != NOT_INTERSECTED && d4 < t2 && d4 != t1)
	{
		t2 = d4;
	}

	return true;
}


inline __device__ void AddIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
	bool first = false;
	bool second = false;

	int tempIndex = p1;

	int list1Index = p1;
	int list2Index = p2;

	float start = NOT_INTERSECTED;


	while (true)
	{
		bool list1 = true;
		bool list2 = true;

		if (list1Index > k1 || sphereIntersections[list1Index] == NOT_INTERSECTED)
		{
			list1 = false;
		}
		if (list2Index > k2 || sphereIntersections[list2Index] == NOT_INTERSECTED)
		{
			list2 = false;
		}
		if (!list1 && !list2) break;


		// skiping not detected intersections
		if (list1 && sphereIntersections[list1Index] == NOT_INTERSECTED)
		{
			list1Index += 2;
			continue;
		}
		if (list2 && sphereIntersections[list2Index] == NOT_INTERSECTED)
		{
			list2Index += 2;
			continue;
		}

		if (list1 && list2)
		{
			list1 = sphereIntersections[list1Index] < sphereIntersections[list2Index];
			list2 = !list1;
		}

		if (list1)
		{

			if (!first && start == NOT_INTERSECTED)
			{
				start = sphereIntersections[list1Index];
			}
			else
			{
				if (!second && first)
				{
					tempArray[tempIndex] = start;
					tempArray[tempIndex + 1] = sphereIntersections[list1Index];
					tempIndex += 2;

					start = NOT_INTERSECTED;
				}
			}

			first = !first;
			list1Index++;
		}
		else
		{

			if (!second && start == NOT_INTERSECTED)
			{
				start = sphereIntersections[list2Index];
			}
			else
			{
				if (!first && second)
				{
					tempArray[tempIndex] = start;
					tempArray[tempIndex + 1] = sphereIntersections[list2Index];
					tempIndex += 2;

					start = NOT_INTERSECTED;
				}
			}

			second = !second;
			list2Index++;
		}
	}

	for (int i = p1; i <= k2; i++)
	{
		if (i < tempIndex)
			sphereIntersections[i] = tempArray[i];
		else
			sphereIntersections[i] = NOT_INTERSECTED;
	}
}


inline __device__ void SubstractIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
	int list1Index = p1;
	int list2Index = p2;
	int addIndex = p1;


	float start1 = sphereIntersections[list1Index];
	float end1 = sphereIntersections[list1Index + 1];
	float start2 = sphereIntersections[list2Index];
	float end2 = sphereIntersections[list2Index + 1];
	while (list1Index <= k1 && list2Index <= k2)
	{
		if (sphereIntersections[list1Index] == NOT_INTERSECTED || sphereIntersections[list2Index] == NOT_INTERSECTED) // one of the lists just ended
		{
			break;
		}

		if (start1 < start2)
		{
			if (end1 < start2) // przedzaily sie nie nakladaja
			{
				tempArray[addIndex] = start1;
				tempArray[addIndex + 1] = end1;
				addIndex += 2;

				list1Index += 2;
				start1 = sphereIntersections[list1Index];
				end1 = sphereIntersections[list1Index + 1];
			}
			else
			{
				if (end1 < end2) // usuwa cala koncowke przedzialu
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = start2;

					addIndex += 2;
					list1Index += 2;
					start1 = sphereIntersections[list1Index];
					end1 = sphereIntersections[list1Index + 1];
				}
				else // wycina przedzial w srodku
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = start2;

					addIndex += 2;

					start1 = end2;

					list2Index += 2;
					start2 = sphereIntersections[list2Index];
					end2 = sphereIntersections[list2Index + 1];
				}
			}

		}
		else
		{
			if (end2 < start1) // brak przeciecia
			{
				list2Index += 2;
				start2 = sphereIntersections[list2Index];
				end2 = sphereIntersections[list2Index + 1];
			}
			else
			{
				if (end2 > end1) // usuwa caly przedzial
				{
					list1Index += 2;
					start1 = sphereIntersections[list1Index];
					end1 = sphereIntersections[list1Index + 1];
				}
				else // usuwa poczatek przedzialu
				{
					start1 = end2;

					list2Index += 2;
					start2 = sphereIntersections[list2Index];
					end2 = sphereIntersections[list2Index + 1];
				}
			}
		}
	}



	if (list2Index > k2 || sphereIntersections[list2Index] == NOT_INTERSECTED)
	{
		if (start1 != end1)
		{
			tempArray[addIndex] = start1;
			tempArray[addIndex + 1] = end1;
			addIndex += 2;
		}
		list1Index += 2;

		while (list1Index <= k1 && sphereIntersections[list1Index] != NOT_INTERSECTED)
		{
			tempArray[addIndex] = sphereIntersections[list1Index];
			tempArray[addIndex + 1] = sphereIntersections[list1Index + 1];
			addIndex += 2;
			list1Index += 2;
		}
	}



	for (int i = p1; i <= k2; i++)
	{
		if (i < addIndex)
			sphereIntersections[i] = tempArray[i];
		else
			sphereIntersections[i] = NOT_INTERSECTED;
	}
}
inline __device__ void CommonPartIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
	int list1Index = p1;
	int list2Index = p2;
	int addIndex = p1;

	while (list1Index < k1 && list2Index < k2)
	{
		if (sphereIntersections[list1Index] == NOT_INTERSECTED || sphereIntersections[list2Index] == NOT_INTERSECTED) // one of the lists just ended
		{
			break;
		}

		float start1 = sphereIntersections[list1Index];
		float end1 = sphereIntersections[list1Index + 1];
		float start2 = sphereIntersections[list2Index];
		float end2 = sphereIntersections[list2Index + 1];

		if (start1 < start2)
		{
			if (end1 < start2)
			{
				list1Index += 2;
			}
			else
			{

				if (end1 < end2)
				{
					tempArray[addIndex] = start2;
					tempArray[addIndex + 1] = end1;
					addIndex += 2;
					list1Index += 2;
				}
				else
				{
					tempArray[addIndex] = start2;
					tempArray[addIndex + 1] = end2;
					addIndex += 2;
					list2Index += 2;
				}
			}
		}
		else
		{
			if (end2 < start1)
			{
				list2Index += 2;
			}
			else
			{
				if (end2 < end1)
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = end2;
					addIndex += 2;
					list2Index += 2;
				}
				else
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = end1;
					addIndex += 2;
					list1Index += 2;
				}
			}
		}
	}
	for (int i = p1; i <= k2; i++)
	{
		if (i < addIndex)
			sphereIntersections[i] = tempArray[i];
		else
			sphereIntersections[i] = NOT_INTERSECTED;
	}
}

template <const int primitiveCount>
__global__ void CalculateInterscetion(
	int width, int height,
	int shape_count,
	CSGNode* nodes,
	CudaPrimitivePos* primitivePos,
	Parameters* primitiveParameters,
	int* parts,
	CudaCamera cam,
	RayHit* hits)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;


	float t1 = NOT_INTERSECTED, t2 = NOT_INTERSECTED;
	float sphereIntersections[2 * primitiveCount]; // 2 floats for each sphere
	float sphereIntersectionsCopy[2 * primitiveCount]; // 2 floats for each sphere
	float tempArray[2 * primitiveCount]; // 2 floats for each sphere

	float3 camera_pos = cam.position;
	//float3 light_pos = make_float3(light_pos_ptr[0], light_pos_ptr[1], light_pos_ptr[2]);

	// Calculate normalized device coordinates (NDC)
	float u = ((float)x + 0.5f) / ((float)width - 1);
	float v = ((float)y + 0.5f) / ((float)height - 1);

	// Convert to screen space coordinates (-1 to 1)
	float nx = (2.0f * u - 1.0f) * ((float)width / (float)height) * tan(cam.fov / 2.0f);
	float ny = (1.0f - 2.0f * v) * tan(cam.fov / 2.0f);

	float3 ray = normalize(nx*cam.right + ny*cam.up + cam.forward);

	for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
	{
		float t1 = NOT_INTERSECTED, t2 = NOT_INTERSECTED;
		if (nodes[k].type == CSGTree::NodeType::Sphere)
		{
			float3 spherePosition = make_float3(primitivePos[nodes[k].primitiveIdx].x, primitivePos[nodes[k].primitiveIdx].y, primitivePos[nodes[k].primitiveIdx].z);
			float radius = primitiveParameters[nodes[k].primitiveIdx].sphereParameters.radius;
			IntersectionPointSphere(spherePosition, radius, camera_pos, ray, t1, t2);
		}
		else if (nodes[k].type == CSGTree::NodeType::Cube)
		{
			if (!IntersectionPointCube(primitivePos[nodes[k].primitiveIdx], primitiveParameters[nodes[k].primitiveIdx], camera_pos, ray, t1, t2))
			{
				t1 = NOT_INTERSECTED;
				t2 = NOT_INTERSECTED;
			}

		}
		else if (nodes[k].type == CSGTree::NodeType::Cylinder)
		{
			if (!IntersectionPointCylinder(primitivePos[nodes[k].primitiveIdx], primitiveParameters[nodes[k].primitiveIdx], camera_pos, ray, t1, t2))
			{
				t1 = NOT_INTERSECTED;
				t2 = NOT_INTERSECTED;
			}
		}

		int m = k - shape_count + 1;


		sphereIntersections[2 * m] = t1;
		sphereIntersections[2 * m + 1] = t2;
		sphereIntersectionsCopy[2 * m] = t1;
		sphereIntersectionsCopy[2 * m + 1] = t2;
	}




	for (int i = shape_count - 2; i >= 0; i--)
	{
		int nodeIndex = i;



		//punkty znajduja sie w lewym od indeksu a do b, w prawym od c do d
		int p1 = parts[4 * nodeIndex];
		int k1 = parts[4 * nodeIndex + 1];
		int p2 = parts[4 * nodeIndex + 2];
		int k2 = parts[4 * nodeIndex + 3];

#ifdef CLASSICRAYCASTING_DEBUG
		if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
		{
			printf("Node %d\n", nodeIndex);
			if (dev_tree.nodes[nodeIndex].type == CSGTree::NodeType::Difference)
			{
				printf("Difference\n");
			}

			else if (dev_tree.nodes[nodeIndex].type == CSGTree::NodeType::Intersection)
			{
				printf("Intersection\n");
			}

			else
			{
				printf("Union\n");
			}
			printf("Operation: %d, p1: %d k1: %d p2: %d k2: %d\n", dev_tree.nodes[nodeIndex].type, p1, k1, p2, k2);
			for (int i = 0; i < 2 * shape_count - 1; i++)
			{
				printf("%f %f ", sphereIntersections[2 * i], sphereIntersections[2 * i + 1]);
			}
			printf("\n");
			printf("\n");
		}
#endif // CLASSICRAYCASTING_DEBUG




		if (nodes[nodeIndex].type == CSGTree::NodeType::Difference)
		{
			SubstractIntervals(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}
		else if (nodes[nodeIndex].type == CSGTree::NodeType::Intersection)
		{
			CommonPartIntervals(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}
		else
		{
			AddIntervals(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}

	}
#ifdef CLASSICRAYCASTING_DEBUG

	if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
	{
		printf("Result\n");
		for (int i = 0; i < 2 * shape_count - 1; i++)
		{
			printf("%f %f ", sphereIntersections[2 * i], sphereIntersections[2 * i + 1]);
		}
		printf("\n");
		printf("\n");
	}
#endif // CLASSICRAYCASTING_DEBUG


	float t = NOT_INTERSECTED;
	for (int i = 0; i <= parts[3]; i++)
	{
		if (sphereIntersections[i] == NOT_INTERSECTED)
			break;
		if (sphereIntersections[i] > 0)
		{
			if (sphereIntersections[i] == sphereIntersections[i + 1])
				continue;
			t = sphereIntersections[i];
			break;
		}
	}

#ifdef CLASSICRAYCASTING_DEBUG

	if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
	{
		printf("t: %f\n", t);
		printf("%f\n", (sphereIntersections[0] - sphereIntersections[1]) * 1000);
		int pixelIdx = (y * (int)width + x);
		RayHit detailedHitInfo;
		detailedHitInfo.hit = false;
		hits[pixelIdx] = detailedHitInfo;
		return;
	}
#endif // CLASSICRAYCASTING_DEBUG

	RayHitMinimal rayHitMinimal;
	rayHitMinimal.hit = CSG::CSGRayHit::Miss;
	rayHitMinimal.t = t;
	if (t > 0 && t != NOT_INTERSECTED && sphereIntersections[0] != sphereIntersections[1])
	{
		for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
		{
			int m = k - shape_count + 1;
			if (t != sphereIntersectionsCopy[2 * m] && t != sphereIntersectionsCopy[2 * m + 1]) continue;

			rayHitMinimal.primitiveIdx = nodes[k].primitiveIdx;
			rayHitMinimal.primitiveType = nodes[k].type;
			rayHitMinimal.hit = (t == sphereIntersectionsCopy[2 * m]) ? CSG::CSGRayHit::Enter : CSG::CSGRayHit::Exit;
			break;
		}
	}

	int pixelIdx = (y * (int)width + x);

	RayHit detailedHitInfo;
	detailedHitInfo.hit = false;
	Ray myRay = Ray(camera_pos, ray);
	if (rayHitMinimal.hit != CSG::CSGRayHit::Miss)
	{
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Sphere)
			sphereHitDetails(myRay, primitivePos[rayHitMinimal.primitiveIdx], primitiveParameters[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Cylinder)
			cylinderHitDetails(myRay, primitivePos[rayHitMinimal.primitiveIdx], primitiveParameters[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Cube)
			cubeHitDetails(myRay, primitivePos[rayHitMinimal.primitiveIdx], primitiveParameters[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
	}
	hits[pixelIdx] = detailedHitInfo;
}

template <const int primitiveCount>
__global__ void CalculateInterscetionShared(
	int width, int height,
	int shape_count,
	CSGNode* nodes,
	CudaPrimitivePos* primitivePos,
	Parameters* primitiveParameters,
	int* parts,
	CudaCamera cam,
	RayHit* hits)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;


	float t1 = NOT_INTERSECTED, t2 = NOT_INTERSECTED;
	float sphereIntersections[2 * primitiveCount]; // 2 floats for each sphere
	float sphereIntersectionsCopy[2 * primitiveCount]; // 2 floats for each sphere
	float tempArray[2 * primitiveCount]; // 2 floats for each sphere

	float3 camera_pos = cam.position;

	// Calculate normalized device coordinates (NDC)
	float u = ((float)x + 0.5f) / ((float)width - 1);
	float v = ((float)y + 0.5f) / ((float)height - 1);

	// Convert to screen space coordinates (-1 to 1)
	float nx = (2.0f * u - 1.0f) * ((float)width / (float)height) * tan(cam.fov / 2.0f);
	float ny = (1.0f - 2.0f * v) * tan(cam.fov / 2.0f);

	float3 ray = normalize(nx * cam.right + ny * cam.up + cam.forward);

	extern __shared__ char shared[];
	CudaPrimitivePos* sharedPrimitivePos = (CudaPrimitivePos*)(shared);
	Parameters* sharedParameters = (Parameters*)(sharedPrimitivePos + CLASSICSHAREDPRIMITIVES);
	short* sharedTypes = (short*)(sharedParameters + CLASSICSHAREDPRIMITIVES);

	for (int shapeProccessed = shape_count - 1; shapeProccessed < 2*shape_count-1; shapeProccessed += CLASSICSHAREDPRIMITIVES)
	{
		int limit = 2 * shape_count - 1 - shapeProccessed < CLASSICSHAREDPRIMITIVES ? 2 * shape_count - 1 - shapeProccessed : CLASSICSHAREDPRIMITIVES;
		for (int i = threadIdx.x + blockDim.x * threadIdx.y; i < limit; i += blockDim.x * blockDim.y)
		{
			short primtiveIdx = nodes[shapeProccessed + i].primitiveIdx;
			sharedTypes[i] = nodes[shapeProccessed + i].type;
			sharedPrimitivePos[i] = primitivePos[primtiveIdx];
            sharedParameters[i] = primitiveParameters[primtiveIdx];
		}
		__syncthreads();
		for (int k = 0; k < limit; k++)
		{
			float t1 = NOT_INTERSECTED, t2 = NOT_INTERSECTED;
			if (sharedTypes[k] == CSGTree::NodeType::Sphere)
			{
				float3 spherePosition = make_float3(sharedPrimitivePos[k].x, sharedPrimitivePos[k].y, sharedPrimitivePos[k].z);
				IntersectionPointSphere(spherePosition, sharedParameters[k].sphereParameters.radius, camera_pos, ray, t1, t2);
			}
			else if (sharedTypes[k] == CSGTree::NodeType::Cube)
			{
				if (!IntersectionPointCube(sharedPrimitivePos[k], sharedParameters[k], camera_pos, ray, t1, t2))
				{
					t1 = NOT_INTERSECTED;
					t2 = NOT_INTERSECTED;
				}

			}
			else if (sharedTypes[k] == CSGTree::NodeType::Cylinder)
			{
				if (!IntersectionPointCylinder(sharedPrimitivePos[k], sharedParameters[k], camera_pos, ray, t1, t2))
				{
					t1 = NOT_INTERSECTED;
					t2 = NOT_INTERSECTED;
				}
			}

			int m = shapeProccessed + k - shape_count + 1;


			sphereIntersections[2 * m] = t1;
			sphereIntersections[2 * m + 1] = t2;
			sphereIntersectionsCopy[2 * m] = t1;
			sphereIntersectionsCopy[2 * m + 1] = t2;
		}
		__syncthreads();
	}

	int4* sharedParts = (int4*)(shared);

	for (int nodesProccessed = shape_count - 2; nodesProccessed >= 0; nodesProccessed -= CLASSICSHAREDPRIMITIVES)
	{
		int limit = nodesProccessed < CLASSICSHAREDPRIMITIVES ? nodesProccessed : CLASSICSHAREDPRIMITIVES;
		for (int i = threadIdx.x + blockDim.x * threadIdx.y; i <= limit; i += blockDim.x * blockDim.y)
		{
			int index = nodesProccessed - i;
			sharedTypes[i] = nodes[index].type;
			sharedParts[i] = ((int4*)(void*)parts)[index];

		}
		__syncthreads();
		for (int k  = 0; k <= limit; k++)
		{
			#ifdef CLASSICRAYCASTING_DEBUG
			if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
			{
				printf("Node %d\n", nodeIndex);
				if (dev_tree.nodes[nodeIndex].type == CSGTree::NodeType::Difference)
				{
					printf("Difference\n");
				}

				else if (dev_tree.nodes[nodeIndex].type == CSGTree::NodeType::Intersection)
				{
					printf("Intersection\n");
				}

				else
				{
					printf("Union\n");
				}
				printf("Operation: %d, p1: %d k1: %d p2: %d k2: %d\n", dev_tree.nodes[nodeIndex].type, p1, k1, p2, k2);
				for (int i = 0; i < 2 * shape_count - 1; i++)
				{
					printf("%f %f ", sphereIntersections[2 * i], sphereIntersections[2 * i + 1]);
				}
				printf("\n");
				printf("\n");
			}
			#endif // CLASSICRAYCASTING_DEBUG




			if (sharedTypes[k] == CSGTree::NodeType::Difference)
			{
				SubstractIntervals(sphereIntersections, tempArray, sharedParts[k].x, sharedParts[k].z, sharedParts[k].y, sharedParts[k].w, false);
			}
			else if (sharedTypes[k] == CSGTree::NodeType::Intersection)
			{
				CommonPartIntervals(sphereIntersections, tempArray, sharedParts[k].x, sharedParts[k].z, sharedParts[k].y, sharedParts[k].w,  false);
			}
			else
			{
				AddIntervals(sphereIntersections, tempArray, sharedParts[k].x, sharedParts[k].z, sharedParts[k].y, sharedParts[k].w, false);
			}

		}
		__syncthreads();
	}
#ifdef CLASSICRAYCASTING_DEBUG

	if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
	{
		printf("Result\n");
		for (int i = 0; i < 2 * shape_count - 1; i++)
		{
			printf("%f %f ", sphereIntersections[2 * i], sphereIntersections[2 * i + 1]);
		}
		printf("\n");
		printf("\n");
	}
#endif // CLASSICRAYCASTING_DEBUG


	float t = NOT_INTERSECTED;
	for (int i = 0; i <= parts[3]; i++)
	{
		if (sphereIntersections[i] == NOT_INTERSECTED)
			break;
		if (sphereIntersections[i] > 0)
		{
			if (sphereIntersections[i] == sphereIntersections[i + 1])
				continue;
			t = sphereIntersections[i];
			break;
		}
	}

#ifdef CLASSICRAYCASTING_DEBUG

	if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
	{
		printf("t: %f\n", t);
		printf("%f\n", (sphereIntersections[0] - sphereIntersections[1]) * 1000);
		int pixelIdx = (y * (int)width + x);
		RayHit detailedHitInfo;
		detailedHitInfo.hit = false;
		hits[pixelIdx] = detailedHitInfo;
		return;
	}
#endif // CLASSICRAYCASTING_DEBUG

	RayHitMinimal rayHitMinimal;
	rayHitMinimal.hit = CSG::CSGRayHit::Miss;
	rayHitMinimal.t = t;
	if (t > 0 && t != NOT_INTERSECTED && sphereIntersections[0] != sphereIntersections[1])
	{
		for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
		{
			int m = k - shape_count + 1;
			if (t != sphereIntersectionsCopy[2 * m] && t != sphereIntersectionsCopy[2 * m + 1]) continue;

			rayHitMinimal.primitiveIdx = nodes[k].primitiveIdx;
			rayHitMinimal.primitiveType = nodes[k].type;
			rayHitMinimal.hit = (t == sphereIntersectionsCopy[2 * m]) ? CSG::CSGRayHit::Enter : CSG::CSGRayHit::Exit;
			break;
		}
	}

	int pixelIdx = (y * (int)width + x);

	RayHit detailedHitInfo;
	detailedHitInfo.hit = false;
	Ray myRay = Ray(camera_pos, ray);
	if (rayHitMinimal.hit != CSG::CSGRayHit::Miss)
	{
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Sphere)
			sphereHitDetails(myRay, primitivePos[rayHitMinimal.primitiveIdx], primitiveParameters[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Cylinder)
			cylinderHitDetails(myRay, primitivePos[rayHitMinimal.primitiveIdx], primitiveParameters[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Cube)
			cubeHitDetails(myRay, primitivePos[rayHitMinimal.primitiveIdx], primitiveParameters[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
	}
	hits[pixelIdx] = detailedHitInfo;
}



inline __device__ bool IntersectionPointSphere(
	float3 spherePosition,
	float radius,
	float3 rayOrigin,
	float3 rayDirection,
	float& t1,
	float& t2)
{
	// Calculate coefficients for the quadratic equation
	float a = dot3(rayDirection, rayDirection);
	float3 rayMinusSphere = make_float3(
		rayOrigin.x - spherePosition.x,
		rayOrigin.y - spherePosition.y,
		rayOrigin.z - spherePosition.z
	);
	float b = 2.0f * dot3(rayDirection, rayMinusSphere);
	float c = dot3(rayMinusSphere, rayMinusSphere) - radius * radius;

	// Calculate discriminant
	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0.0f)
	{
		return false; // No intersection
	}

	// Calculate t1 and t2 (solutions to the quadratic equation)
	float sqrtDiscriminant = sqrt(discriminant);
	t1 = (-b - sqrtDiscriminant) / (2.0f * a);
	t2 = (-b + sqrtDiscriminant) / (2.0f * a);

	return true; // Intersection found
}



inline __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix)
{
	float4 result = { 0, 0, 0, 0 };
	result.x = vector.x * matrix[0] + vector.y * matrix[1] + vector.z * matrix[2] + vector.w * matrix[3];
	result.y = vector.x * matrix[4] + vector.y * matrix[5] + vector.z * matrix[6] + vector.w * matrix[7];
	result.z = vector.x * matrix[8] + vector.y * matrix[9] + vector.z * matrix[10] + vector.w * matrix[11];
	result.w = vector.x * matrix[12] + vector.y * matrix[13] + vector.z * matrix[14] + vector.w * matrix[15];

	vector = result;
}

inline __device__ float4 NormalizeVector4(float4 vector)
{
	float length = sqrt(vector.x * vector.x +
		vector.y * vector.y +
		vector.z * vector.z +
		vector.w * vector.w);

	vector.x /= length;
	vector.y /= length;
	vector.z /= length;
	vector.w /= length;

	return vector;
}


inline __device__ float3 NormalizeVector3(float3 vector)
{
	float length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	vector.x /= length;
	vector.y /= length;
	vector.z /= length;
	return vector;
}

inline __device__ float dot3(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross(float3 a, float3 b)
{
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}