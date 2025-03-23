#include "RaycastingKernels.cuh"

__global__ void RaycastKernel(Camera cam, CudaCSGTree tree, RayHit* hits, float width, float height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	// Calculate normalized device coordinates (NDC)
	float u = ((float)x+0.5f) / (width - 1);
	float v = ((float)y + 0.5f) / (height - 1);

	// Convert to screen space coordinates (-1 to 1)
	float nx = (2.0f * u - 1.0f) * (width / height) * tan(cam.fov / 2.0f);
	float ny = (1.0f - 2.0f * v) * tan(cam.fov / 2.0f);


	// Create ray from camera
	float3 rayOrigin = make_float3(cam.x, cam.y, cam.z);
	float3 rayDirection = normalize(make_float3(
		cam.right[0] * nx + cam.up[0] * ny + cam.forward[0],
		cam.right[1] * nx + cam.up[1] * ny + cam.forward[1],
		cam.right[2] * nx + cam.up[2] * ny + cam.forward[2]
	));

	Ray ray(rayOrigin, rayDirection);
	RayHitMinimal hitInfo;

	// For now, just test against the first sphere
	
	CSGRayCast(tree, ray, hitInfo);
	int pixelIdx = (y * (int)width + x);
	
	RayHit detailedHitInfo;
	detailedHitInfo.hit = false;
	if (hitInfo.hit != CSG::CSGRayHit::Miss)
	{
		if(hitInfo.primitiveType == CSGTree::NodeType::Sphere)
			sphereHitDetails(ray, tree.primitives[hitInfo.primitiveIdx], hitInfo, detailedHitInfo);
		if (hitInfo.primitiveType == CSGTree::NodeType::Cylinder)
			cylinderHitDetails(ray, tree.primitives[hitInfo.primitiveIdx], hitInfo, detailedHitInfo);
		if (hitInfo.primitiveType == CSGTree::NodeType::Cube)
			cubeHitDetails(ray, tree.primitives[hitInfo.primitiveIdx], hitInfo, detailedHitInfo);
	}
	hits[pixelIdx] = detailedHitInfo;
}

__global__ void LightningKernel(Camera cam, RayHit* hits, Primitive* primitives, float4* output, float3 lightDir ,float width, float height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIdx = (y * (int)width + x);

	RayHit hitInfo = hits[pixelIdx];



	if (hitInfo.hit)
	{
		float3 color = make_float3(primitives[hitInfo.primitiveIdx].r,
			primitives[hitInfo.primitiveIdx].g,
			primitives[hitInfo.primitiveIdx].b);

		float3 lightColor = make_float3(1.0f, 1.0f, 1.0f); 

		
		float ka = 0.2f;   
		float kd = 0.8f;    
		float ks = 0.7f;    
		float shininess = 30.0f; 

		// Calculate lighting vectors

		lightDir = normalize(lightDir);
		float3 viewDir = normalize(make_float3(cam.x, cam.y, cam.z) - hitInfo.position);
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

__device__ void hitPrimitive(const Ray& ray, const CudaCSGTree& tree, const CSGNode& node, RayHitMinimal& hitInfo, float& tmin)
{
	if (node.type == CSGTree::NodeType::Sphere)
	{
		sphereHit(ray, tree.primitives[node.primitiveIdx], hitInfo, tmin);
	}
	else if (node.type == CSGTree::NodeType::Cylinder)
	{
		cylinderHit(ray, tree.primitives[node.primitiveIdx], hitInfo, tmin);
	}
	else if (node.type == CSGTree::NodeType::Cube)
	{
		cubeHit(ray, tree.primitives[node.primitiveIdx], hitInfo, tmin);
	}
	else
	{
		hitInfo = RayHitMinimal();
	}


}

__device__ bool sphereHit(const Ray& ray, const Primitive& sphere, RayHitMinimal& hitInfo, float& tmin)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	hitInfo.primitiveIdx = sphere.id;
	float3 oc = make_float3(
		ray.origin.x - sphere.x,
		ray.origin.y - sphere.y,
		ray.origin.z - sphere.z
	);

	float b = dot(oc, ray.direction);
	float c = dot(oc, oc) - sphere.params.sphereParameters.radius * sphere.params.sphereParameters.radius;
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
			normal.x - sphere.x,
			normal.y - sphere.y,
			normal.z - sphere.z);
	
	if (dot(normal, ray.direction) <= 0)
		hitInfo.hit = CSG::CSGRayHit::Enter;
	else
		hitInfo.hit = CSG::CSGRayHit::Exit;

	hitInfo.primitiveType = CSGTree::NodeType::Sphere;

	return true;
}

__device__ void sphereHitDetails(const Ray& ray, const Primitive& sphere, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.t = hitInfo.t;
	detailedHitInfo.position = ray.computePosition(detailedHitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;
	detailedHitInfo.normal = normalize(
		make_float3(
			detailedHitInfo.position.x - sphere.x,
			detailedHitInfo.position.y - sphere.y,
			detailedHitInfo.position.z - sphere.z
		)
	);
	if (hitInfo.hit & CSG::CSGRayHit::Flip)
		detailedHitInfo.normal = -detailedHitInfo.normal;
	if (hitInfo.hit & CSG::CSGRayHit::Exit)
		detailedHitInfo.normal = -detailedHitInfo.normal;
}

__device__ bool cylinderHit(const Ray& ray, const Primitive& cylinder, RayHitMinimal& hitInfo, float& tmin)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	hitInfo.primitiveIdx = cylinder.id;
	CylinderParameters params = cylinder.params.cylinderParameters;
	float3 V = make_float3(params.axisX, params.axisY, params.axisZ);
	float3 C = make_float3(cylinder.x, cylinder.y, cylinder.z) - (cylinder.params.cylinderParameters.height / 2) * V;
	float3 OC = ray.origin - C;

	float aHelp = dot(ray.direction, V);
	float a = fmax(1 - aHelp*aHelp, 0.00001f);

	float cHelp = dot(OC, V);
	float c = dot(OC, OC) - cHelp * cHelp - params.radius * params.radius;

	float b =  (dot(ray.direction, OC) - dot(ray.direction, V) * dot(OC, V));
	float discriminant = b * b -  a * c;

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
			float3 OCmMax = ray.origin - make_float3(cylinder.x, cylinder.y, cylinder.z) - (cylinder.params.cylinderParameters.height / 2) * V;
			temp = dot(-OCmMax, V) / den;
			surfNormal = V;
			useSurfNormal = 2;
		}
	}
	if (temp <= tmin||skip)
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
				float3 OCmMax = ray.origin - make_float3(cylinder.x, cylinder.y, cylinder.z) - (cylinder.params.cylinderParameters.height / 2) * V;
				temp = dot(-OCmMax, V) / den;
				surfNormal =  V;
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

__device__ void cylinderHitDetails(const Ray& ray, const Primitive& cylinder, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.t = hitInfo.t;
	detailedHitInfo.position = ray.computePosition(detailedHitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;

	CylinderParameters params = cylinder.params.cylinderParameters;
	float3 V = make_float3(params.axisX, params.axisY, params.axisZ);
	float3 C = make_float3(cylinder.x, cylinder.y, cylinder.z) - (cylinder.params.cylinderParameters.height / 2) * V;
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
__device__ bool cubeHit(const Ray& ray, const Primitive& cube, RayHitMinimal& hitInfo, float& tmin)
{
	hitInfo.hit = CSG::CSGRayHit::Miss;
	hitInfo.primitiveIdx = cube.id;



	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray

	float3 axis = make_float3(1, 1, 1);
	float3 C = make_float3(cube.x, cube.y, cube.z);
	float3 lb = C - cube.params.cubeParameters.size / 2 * axis;
	float3 rt = C + cube.params.cubeParameters.size / 2 * axis;
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
	float halfSize = cube.params.cubeParameters.size / 2;
	float3 normal = make_float3((float)(int)(PC.x / halfSize * bias), (float)(int)(PC.y / halfSize * bias), (float)(int)(PC.z / halfSize * bias));


	if (dot(normal, ray.direction) <= 0)
		hitInfo.hit = CSG::CSGRayHit::Enter;
	else
		hitInfo.hit = CSG::CSGRayHit::Exit;

	hitInfo.primitiveType = CSGTree::NodeType::Cube;
	return true;
}

__device__ void cubeHitDetails(const Ray& ray, const Primitive& cube, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo)
{
	detailedHitInfo.hit = true;
	detailedHitInfo.t = hitInfo.t;
	detailedHitInfo.position = ray.computePosition(detailedHitInfo.t);
	detailedHitInfo.primitiveIdx = hitInfo.primitiveIdx;

	float3 C = make_float3(cube.x, cube.y, cube.z);


	float3 PC = detailedHitInfo.position - C;
	float bias = 1.00001;
	float halfSize = cube.params.cubeParameters.size / 2;
	float3 normal = make_float3((float)(int)(PC.x / halfSize * bias), (float)(int)(PC.y / halfSize * bias), (float)(int)(PC.z / halfSize * bias));

	detailedHitInfo.normal = normalize(normal);

	if (hitInfo.hit & CSG::CSGRayHit::Flip)
		detailedHitInfo.normal = -detailedHitInfo.normal;
	if (hitInfo.hit & CSG::CSGRayHit::Exit)
		detailedHitInfo.normal = -detailedHitInfo.normal;
}

__device__ void CSGRayCast(CudaCSGTree& tree, Ray& ray, RayHitMinimal& resultRayhit)
{

	CudaStack<unsigned char, MAXSTACKSIZE> actionStack;
	CudaStack<RayHitMinimal, MAXSTACKSIZE> primitiveStack;
	CudaStack<float, MAXSTACKSIZE> timeStack;

	float tmin = 0;
	CSGNode node = CSGNode(0, 0, 0, 0, 0);
	RayHitMinimal leftRay;
	RayHitMinimal rightRay;
	actionStack.push(CSG::CSGActions::Compute);
	unsigned char action = CSG::CSGActions::GotoLft;
	bool run = true;

	while (run||actionStack.size()>0)
	{
		if (action & CSG::CSGActions::SaveLft)
		{
			tmin = timeStack.pop();
			primitiveStack.push(leftRay);
			action = CSG::CSGActions::GotoRgh;
		}
		if (action & (CSG::CSGActions::GotoLft | CSG::CSGActions::GotoRgh))
		{
			GoTo(actionStack,
				 primitiveStack,
				 timeStack,
				 action,
				 node,
				 tree,
				 leftRay,
				 rightRay,
				 ray,
				 tmin, 
				 run);
		}
		if (action & (CSG::CSGActions::LoadLft | CSG::CSGActions::LoadRgh | CSG::CSGActions::Compute))
		{
			Compute(actionStack,
				primitiveStack,
				timeStack,
				action,
				node,
				tree,
				leftRay,
				rightRay,
				tmin,
				run);
		}
	}

	resultRayhit = leftRay;
}

__device__ void GoTo(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	CudaCSGTree& tree,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	Ray& ray,
	float& tmin,
	bool& run)
{
	if (action & CSG::CSGActions::GotoLft)
	{
		node = tree.nodes[node.left];
	}
	else
	{
		node = tree.nodes[node.right];
	}

	if (node.type == CSGTree::NodeType::Union ||
		node.type == CSGTree::NodeType::Difference ||
		node.type == CSGTree::NodeType::Intersection)
	{
		bool gotoL = isBVHNodeHit(ray, tree.nodes[node.left].bvhNode, leftRay,tmin);
		bool gotoR = isBVHNodeHit(ray, tree.nodes[node.right].bvhNode, rightRay,tmin);
		CSGNode tmpNode = tree.nodes[node.left];
		if (gotoL && (tmpNode.primitiveIdx != -1))
		{
			hitPrimitive(ray, tree, tmpNode, leftRay, tmin);
			gotoL = false;
		}
		tmpNode = tree.nodes[node.right];
		if (gotoR && (tree.nodes[node.right].primitiveIdx != -1))
		{
			hitPrimitive(ray, tree, tmpNode, rightRay, tmin);
			gotoR = false;
		}
		if (gotoL || gotoR)
		{
			if (!gotoL)
			{
				primitiveStack.push(leftRay);
				actionStack.push(CSG::CSGActions::LoadLft);
				action = CSG::CSGActions::GotoRgh;
			}
			else if (!gotoR)
			{
				primitiveStack.push(rightRay);
				actionStack.push(CSG::CSGActions::LoadRgh);
				action = CSG::CSGActions::GotoLft;
			}
			else
			{
				timeStack.push(tmin);
				actionStack.push(CSG::CSGActions::LoadLft);
				actionStack.push(CSG::CSGActions::SaveLft);
				action = CSG::CSGActions::GotoLft;
			}
		}
		else
		{
			action = CSG::CSGActions::Compute;
		}

	}
	else
	{
		if (action & CSG::CSGActions::GotoLft)
		{
			hitPrimitive(ray, tree, node, leftRay, tmin);
		}
		else
		{
			hitPrimitive(ray, tree, node, rightRay, tmin);
		}
		action = actionStack.pop();
		node = GetParent(tree, node, run);
	}
}

__device__ void Compute(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	CudaCSGTree& tree,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	float& tmin, 
	bool& run)
{
	if (action & (CSG::CSGActions::LoadLft | CSG::CSGActions::LoadRgh))
	{
		if (action & CSG::CSGActions::LoadLft)
		{
			leftRay = primitiveStack.pop();
		}
		else
		{
			rightRay = primitiveStack.pop();
		}
	}
	int actions = LookUpActions(leftRay.hit, rightRay.hit, node.type);
	if ((actions & CSG::HitActions::RetL) || ((actions & CSG::HitActions::RetLIfCloser) && (leftRay.t < rightRay.t)))
	{
		rightRay = leftRay;
		action = actionStack.pop();
		node = GetParent(tree, node, run);
	}
	else if ((actions & CSG::HitActions::RetR) || ((actions & CSG::HitActions::RetRIfCloser) && (leftRay.t > rightRay.t)))
	{
		if (actions & CSG::HitActions::FlipR)
		{
			rightRay.hit ^= CSG::CSGRayHit::Flip;
			rightRay.hit ^= CSG::CSGRayHit::Exit;
			rightRay.hit ^= CSG::CSGRayHit::Enter;
			
		}
		leftRay = rightRay;
		action = actionStack.pop();
		node = GetParent(tree, node, run);
	}
	else if ((actions & CSG::HitActions::LoopL) || ((actions & CSG::HitActions::LoopLIfCloser) && (leftRay.t < rightRay.t)))
	{
		tmin = leftRay.t;
		primitiveStack.push(rightRay);
		actionStack.push(CSG::CSGActions::LoadRgh);
		action = CSG::CSGActions::GotoLft;
	}
	else if ((actions & CSG::HitActions::LoopR) || ((actions & CSG::HitActions::LoopRIfCloser) && (leftRay.t > rightRay.t)))
	{
		tmin = rightRay.t;
		primitiveStack.push(leftRay);
		actionStack.push(CSG::CSGActions::LoadLft);
		action = CSG::CSGActions::GotoRgh;
	}
	else
	{
		rightRay = RayHitMinimal();
		leftRay = RayHitMinimal();
		action = actionStack.pop();
		node = GetParent(tree, node, run);
	}
}


__device__ int LookUpActions(unsigned char lHit, unsigned char rHit, int op)
{
	static int unionTable[3][3] = {
	{{CSG::HitActions::RetLIfCloser | CSG::HitActions::RetRIfCloser},{CSG::HitActions::RetRIfCloser | CSG::HitActions::LoopL},{CSG::HitActions::RetL}},
	{{CSG::HitActions::RetLIfCloser | CSG::HitActions::LoopR},{CSG::HitActions::LoopLIfCloser | CSG::HitActions::LoopRIfCloser},{CSG::HitActions::RetL}},
	{{CSG::HitActions::RetR},{CSG::HitActions::RetR},{CSG::HitActions::MissAction}} };
	static int intersectionTable[3][3] = {
	{{CSG::HitActions::LoopLIfCloser | CSG::HitActions::LoopRIfCloser},{CSG::HitActions::RetLIfCloser|CSG::HitActions::LoopR},{CSG::HitActions::MissAction}},
	{{CSG::HitActions::RetRIfCloser | CSG::HitActions::LoopL},{CSG::HitActions::RetLIfCloser | CSG::HitActions::RetRIfCloser},{CSG::HitActions::MissAction}},
	{{CSG::HitActions::MissAction},{CSG::HitActions::MissAction},{CSG::HitActions::MissAction}} };
	static int differenceTable[3][3] = {
	{{CSG::HitActions::RetLIfCloser | CSG::HitActions::LoopR},{CSG::HitActions::LoopLIfCloser | CSG::HitActions::LoopRIfCloser},{CSG::HitActions::RetL}},
	{{CSG::HitActions::RetLIfCloser | CSG::HitActions::RetRIfCloser |CSG::HitActions::FlipR},{CSG::HitActions::RetRIfCloser | CSG::HitActions::FlipR|CSG::HitActions::LoopL},{CSG::HitActions::RetL}},
	{{CSG::HitActions::MissAction},{CSG::HitActions::MissAction},{CSG::HitActions::MissAction}} };

	if (lHit & CSG::CSGRayHit::Enter)
		lHit = 0;
	if (lHit & CSG::CSGRayHit::Exit)
		lHit = 1;
	if (lHit & CSG::CSGRayHit::Miss)
		lHit = 2;

	if (rHit & CSG::CSGRayHit::Enter)
		rHit = 0;
	if (rHit & CSG::CSGRayHit::Exit)
		rHit = 1;
	if (rHit & CSG::CSGRayHit::Miss)
		rHit = 2;

	if (op == CSGTree::NodeType::Union)
	{
		return unionTable[lHit][rHit];
	}
	if (op == CSGTree::NodeType::Intersection)
	{
		return intersectionTable[lHit][rHit];
	}
	if (op == CSGTree::NodeType::Difference)
	{
		return differenceTable[lHit][rHit];
	}
	return -1;
}

__device__ CSGNode GetParent(CudaCSGTree& tree, CSGNode& node, bool& run)
{
	if (node.parent >= 0)
	{
		return tree.nodes[node.parent];
	}
	run = false;
	return CSGNode(0, 0, 0, 0, 0);
}

__device__ bool isBVHNodeHit(const Ray& ray, const BVHNode& node, RayHitMinimal& hitInfo, float& tmin)
{


	float3 lb = make_float3(node.minX, node.minY, node.minZ);
	float3 rt = make_float3(node.maxX, node.maxY, node.maxZ);
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
		hitInfo.hit = CSG::CSGRayHit::Miss;
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tempmin > tempmax)
	{
		hitInfo.hit = CSG::CSGRayHit::Miss;
		return false;
	}
	if (tempmin <= tmin)
	{
		if (tempmax <= tmin)
		{
			hitInfo.hit = CSG::CSGRayHit::Miss;
			return false;
		}
	}

	return true;
}

#define NOT_INTERSECTED 1000

#define DEBUG_PIXEL_X 300
#define DEBUG_PIXEL_Y 300



__device__ bool IntersectionPointCube(const Primitive& cube, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2)
{
	float3 axis = make_float3(1, 1, 1);
	float3 C = make_float3(cube.x, cube.y, cube.z);
	float3 l = C - cube.params.cubeParameters.size / 2 * axis;
	float3 h = C + cube.params.cubeParameters.size / 2 * axis;
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

	if (t_close == tx_close)
	{
		if (r.x > 0)
			N = make_float3(-1, 0, 0);
		else
			N = make_float3(1, 0, 0);
	}
	if (t_close == ty_close)
	{
		if (r.y > 0)
			N = make_float3(0, -1, 0);
		else
			N = make_float3(0, 1, 0);
	}
	if (t_close == tz_close)
	{
		if (r.z > 0)
			N = make_float3(0, 0, -1);
		else
			N = make_float3(0, 0, 1);
	}


	if (t_far == tx_far)
	{
		if (r.x > 0)
			N2 = make_float3(-1, 0, 0);
		else
			N2 = make_float3(1, 0, 0);
	}
	if (t_far == ty_far)
	{
		if (r.y > 0)
			N2 = make_float3(0, -1, 0);
		else
			N2 = make_float3(0, 1, 0);
	}
	if (t_far == tz_far)
	{
		if (r.z > 0)
			N2 = make_float3(0, 0, -1);
		else
			N2 = make_float3(0, 0, 1);
	}

	t1 = t_close;
	t2 = t_far;

	return t_close < t_far;
}

__device__ float3 CalculateNormalVectorCylinder(const Primitive& cylinder, float3 pixelPosition)
{
	float3 axis = make_float3(cylinder.params.cylinderParameters.axisX, cylinder.params.cylinderParameters.axisY, cylinder.params.cylinderParameters.axisZ);
	float t = dot3(make_float3(pixelPosition.x - cylinder.x, pixelPosition.y - cylinder.y, pixelPosition.z - cylinder.z), axis);
	float3 Cp = make_float3(cylinder.x + t * cylinder.x, cylinder.y + t * cylinder.y, cylinder.z + t * cylinder.z);
	float3 r = make_float3(pixelPosition.x - Cp.x, pixelPosition.y - Cp.y, pixelPosition.z - Cp.z);
	return NormalizeVector3(r);
}

__device__ bool IntersectionPointCylinder(const Primitive& cylinder, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2)
{
	float3 axis = make_float3(cylinder.params.cylinderParameters.axisX, cylinder.params.cylinderParameters.axisY, cylinder.params.cylinderParameters.axisZ);
	float3 b = make_float3(cylinder.x - rayOrigin.x, cylinder.y - rayOrigin.y, cylinder.z - rayOrigin.z) - (cylinder.params.cylinderParameters.height / 2) * axis;
	float3 a = NormalizeVector3(axis);
	float r = cylinder.params.cylinderParameters.radius;
	float h = cylinder.params.cylinderParameters.height;
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

	if (dot3(make_float3(n.x * d3 - c2.x, n.y * d3 - c2.y, n.z * d3 - c2.z), make_float3(n.x * d3 - c2.x, n.y * d3 - c2.y, n.z * d3 - c2.z)) > r * r || d3 < 0)
		d3 = NOT_INTERSECTED;
	if (dot3(make_float3(n.x * d4 - c1.x, n.y * d4 - c1.y, n.z * d4 - c1.z), make_float3(n.x * d4 - c1.x, n.y * d4 - c1.y, n.z * d4 - c1.z)) > r * r || d4 < 0)
		d4 = NOT_INTERSECTED;

	t1 = NOT_INTERSECTED;
	t2 = NOT_INTERSECTED;
	if (d1 != NOT_INTERSECTED)
	{
		t1 = d1;
		N = CalculateNormalVectorCylinder(cylinder, make_float3(rayOrigin.x + t1 * rayDirection.x, rayOrigin.y + t1 * rayDirection.y, rayOrigin.z + t1 * rayDirection.z));
	}
	if (d3 != NOT_INTERSECTED && d3 < t1)
	{
		t1 = d3;
		N = axis;
	}
	if (d4 != NOT_INTERSECTED && d4 < t1)
	{
		t1 = d4;
		N = make_float3(-axis.x, -axis.y, -axis.z);
	}

	// finding smallest t2
	if (d2 != NOT_INTERSECTED)
	{
		t2 = d2;
		N2 = CalculateNormalVectorCylinder(cylinder, make_float3(rayOrigin.x + t2 * rayDirection.x, rayOrigin.y + t2 * rayDirection.y, rayOrigin.z + t2 * rayDirection.z));
		N2 = make_float3(-N2.x, -N2.y, -N2.z);
	}
	if (d3 != NOT_INTERSECTED && d3 < t2 && d3 != t1)
	{
		t2 = d3;
		N2 = make_float3(-axis.x, -axis.y, -axis.z);
	}
	if (d4 != NOT_INTERSECTED && d4 < t2 && d4 != t1)
	{
		t2 = d4;
		N2 = axis;
	}

	return true;
}

__device__ void AddIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
	// merging two lists into tempArray sorted by start time
	int list1Index = p1;
	int list2Index = p2;
	int tempIndex = p1;
	while (list1Index < k1 && list2Index < k2)
	{
		if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
		{
			break;
		}

		if (sphereIntersections[list2Index] < sphereIntersections[list1Index])
		{
			tempArray[tempIndex] = sphereIntersections[list2Index];
			tempArray[tempIndex + 1] = sphereIntersections[list2Index + 1];
			list2Index += 2;
		}
		else
		{
			tempArray[tempIndex] = sphereIntersections[list1Index];
			tempArray[tempIndex + 1] = sphereIntersections[list1Index + 1];
			list1Index += 2;
		}
		tempIndex += 2;
	}
	while (list1Index < k1 && sphereIntersections[list1Index] != -1)
	{
		tempArray[tempIndex] = sphereIntersections[list1Index];
		tempArray[tempIndex + 1] = sphereIntersections[list1Index + 1];
		list1Index += 2;
		tempIndex += 2;
	}
	while (list2Index < k2 && sphereIntersections[list2Index] != -1)
	{
		tempArray[tempIndex] = sphereIntersections[list2Index];
		tempArray[tempIndex + 1] = sphereIntersections[list2Index + 1];
		list2Index += 2;
		tempIndex += 2;
	}



	// merging tempArray into sphereIntersections
	if (tempIndex != p1) //if something changed
	{
		float start = tempArray[p1];
		float end = tempArray[p1 + 1];
		int addIndex = p1;
		for (int i = p1 + 2; i <= tempIndex - 2; i += 2)
		{
			float currentStart = tempArray[i];
			float currentEnd = tempArray[i + 1];
			if (currentStart > end)
			{
				sphereIntersections[addIndex] = start;
				sphereIntersections[addIndex + 1] = end;
				addIndex += 2;
				start = currentStart;
				end = currentEnd;
			}
			else
			{
				if (currentEnd > end)
					end = currentEnd;
			}
		}
		sphereIntersections[addIndex] = start;
		sphereIntersections[addIndex + 1] = end;
		addIndex += 2;


		for (int i = addIndex; i <= k2; i++)
		{
			sphereIntersections[i] = -1;
		}
	}
}

__host__ __device__ void AddIntervals2(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
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

		if (list1Index > k1 || sphereIntersections[list1Index] == -1)
		{
			list1 = false;
		}
		if (list2Index > k2 || sphereIntersections[list2Index] == -1)
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
			sphereIntersections[i] = -1;
	}
}


__host__ __device__ void SubstractIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
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
		if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
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



	if (list2Index > k2 || sphereIntersections[list2Index] == -1)
	{
		if (start1 != end1)
		{
			tempArray[addIndex] = start1;
			tempArray[addIndex + 1] = end1;
			addIndex += 2;
		}
		list1Index += 2;

		while (list1Index <= k1 && sphereIntersections[list1Index] != -1)
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
			sphereIntersections[i] = -1;
	}

}
__host__ __device__ void CommonPartIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
	int list1Index = p1;
	int list2Index = p2;
	int addIndex = p1;

	while (list1Index < k1 && list2Index < k2)
	{
		if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
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
			sphereIntersections[i] = -1;
	}

}


__global__ void CalculateInterscetion(int width, int height, int shape_count, CudaCSGTree dev_tree, int* parts, Camera cam,
	RayHit* hits)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;


	float t1 = -1, t2 = -1;
	const int sphereCount = 256;
	float sphereIntersections[2 * sphereCount]; // 2 floats for each sphere
	float sphereIntersectionsCopy[2 * sphereCount]; // 2 floats for each sphere
	float3 normalVectors[2 * sphereCount]; // 2 floats for each sphere
	float tempArray[2 * sphereCount]; // 2 floats for each sphere

	float3 camera_pos = make_float3(cam.x, cam.y, cam.z);
	//float3 light_pos = make_float3(light_pos_ptr[0], light_pos_ptr[1], light_pos_ptr[2]);

	// Calculate normalized device coordinates (NDC)
	float u = ((float)x + 0.5f) / ((float)width - 1);
	float v = ((float)y + 0.5f) / ((float)height - 1);

	// Convert to screen space coordinates (-1 to 1)
	float nx = (2.0f * u - 1.0f) * ((float)width / (float)height) * tan(cam.fov / 2.0f);
	float ny = (1.0f - 2.0f * v) * tan(cam.fov / 2.0f);

	float3 ray = normalize(make_float3(
		cam.right[0] * nx + cam.up[0] * ny + cam.forward[0],
		cam.right[1] * nx + cam.up[1] * ny + cam.forward[1],
		cam.right[2] * nx + cam.up[2] * ny + cam.forward[2]
	));

	for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
	{
		float t1 = -1, t2 = -1;
		float3 N1, N2;
		if (dev_tree.nodes[k].type == CSGTree::NodeType::Sphere)
		{
			float3 spherePosition = make_float3(dev_tree.primitives[dev_tree.nodes[k].primitiveIdx].x, dev_tree.primitives[dev_tree.nodes[k].primitiveIdx].y, dev_tree.primitives[dev_tree.nodes[k].primitiveIdx].z);
			float radius = dev_tree.primitives[dev_tree.nodes[k].primitiveIdx].params.sphereParameters.radius;
			IntersectionPointSphere(spherePosition, radius, camera_pos, ray, t1, t2);

			float3 pixelPosition1 = make_float3(camera_pos.x + t1 * ray.x, camera_pos.y + t1 * ray.y, camera_pos.z + t1 * ray.z);
			float3 pixelPosition2 = make_float3(camera_pos.x + t2 * ray.x, camera_pos.y + t2 * ray.y, camera_pos.z + t2 * ray.z);

			N1 = NormalizeVector3(make_float3(pixelPosition1.x - spherePosition.x, pixelPosition1.y - spherePosition.y, pixelPosition1.z - spherePosition.z));
			N2 = NormalizeVector3(make_float3(-pixelPosition2.x + spherePosition.x, -pixelPosition2.y + spherePosition.y, -pixelPosition2.z + spherePosition.z));
		}
		else if (dev_tree.nodes[k].type == CSGTree::NodeType::Cube)
		{
			Primitive cube = dev_tree.primitives[dev_tree.nodes[k].primitiveIdx];
			if (!IntersectionPointCube(cube, camera_pos, ray, t1, t2, N1, N2))
			{
				t1 = -1;
				t2 = -1;
			}

		}
		else if (dev_tree.nodes[k].type == CSGTree::NodeType::Cylinder)
		{
			Primitive cylinder = dev_tree.primitives[dev_tree.nodes[k].primitiveIdx];

			if (!IntersectionPointCylinder(cylinder, camera_pos, ray, t1, t2, N1, N2))
			{
				t1 = -1;
				t2 = -1;
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


		/*if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
		{
			printf("Node %d\n", nodeIndex);
			printf("Operation: %c, p1: %d k1: %d p2: %d k2: %d\n", dev_tree[nodeIndex].operation, p1, k1, p2, k2);
			for (int i = 0; i < 2 * shape_count - 1; i++)
			{
				printf("%f %f ", sphereIntersections[2 * i], sphereIntersections[2 * i + 1]);
			}
			printf("\n");
			printf("\n");
		}*/


		if (dev_tree.nodes[nodeIndex].type == CSGTree::NodeType::Difference)
		{
			SubstractIntervals(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}

		else if (dev_tree.nodes[nodeIndex].type == CSGTree::NodeType::Intersection)
		{
			CommonPartIntervals(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}

		else
		{
			AddIntervals2(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}

	}



	float t = sphereIntersections[0];

	/*if (DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y)
	{

		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = 255;
		dev_texture_data[index2 + 1] = 255;
		dev_texture_data[index2 + 2] = 0;
		return;
	}


	if (t < 0 || sphereIntersections[0] == sphereIntersections[1])
	{
		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = 0;
		dev_texture_data[index2 + 1] = 0;
		dev_texture_data[index2 + 2] = 0;

		return;
	}*/

	RayHitMinimal rayHitMinimal;
	rayHitMinimal.hit = CSG::CSGRayHit::Miss;
	rayHitMinimal.t = t;
	if (t >0)
	{
		for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
		{
			int m = k - shape_count + 1;
			if (t != sphereIntersectionsCopy[2 * m] && t != sphereIntersectionsCopy[2 * m + 1]) continue;

			rayHitMinimal.primitiveIdx = dev_tree.nodes[k].primitiveIdx;
			rayHitMinimal.primitiveType = dev_tree.nodes[k].type;
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
			sphereHitDetails(myRay, dev_tree.primitives[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Cylinder)
			cylinderHitDetails(myRay, dev_tree.primitives[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
		if (rayHitMinimal.primitiveType == CSGTree::NodeType::Cube)
			cubeHitDetails(myRay, dev_tree.primitives[rayHitMinimal.primitiveIdx], rayHitMinimal, detailedHitInfo);
	}
	hits[pixelIdx] = detailedHitInfo;
}

__device__ bool IntersectionPointSphere(
	const float3& spherePosition,
	float radius,
	const float3& rayOrigin,
	const float3& rayDirection,
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



__host__ __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix)
{
	float4 result = { 0, 0, 0, 0 };
	result.x = vector.x * matrix[0] + vector.y * matrix[1] + vector.z * matrix[2] + vector.w * matrix[3];
	result.y = vector.x * matrix[4] + vector.y * matrix[5] + vector.z * matrix[6] + vector.w * matrix[7];
	result.z = vector.x * matrix[8] + vector.y * matrix[9] + vector.z * matrix[10] + vector.w * matrix[11];
	result.w = vector.x * matrix[12] + vector.y * matrix[13] + vector.z * matrix[14] + vector.w * matrix[15];

	vector = result;
}

__host__ __device__ float4 NormalizeVector4(float4 vector)
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


__host__ __device__ float3 NormalizeVector3(float3 vector)
{
	float length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	vector.x /= length;
	vector.y /= length;
	vector.z /= length;
	return vector;
}

__host__ __device__ float dot3(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(const float3& a, const float3& b)
{
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}
