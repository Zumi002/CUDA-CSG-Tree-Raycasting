#include "RaycastingKernels.cuh"

__global__ void RaycastKernel(Camera cam, CudaCSGTree tree, RayHit* hits, float width, float height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	// Calculate normalized device coordinates (NDC)
	float u = (float)x / (width - 1);
	float v = (float)y / (height - 1);

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
			temp = dot(-OC+params.height* V, V) / den;
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
				temp = dot(-OC + params.height * V, V) / den;
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

	float3 dirfrac = make_float3(0, 0, 0);
	dirfrac.x = 1.0f / ray.direction.x;
	dirfrac.y = 1.0f / ray.direction.y;
	dirfrac.z = 1.0f / ray.direction.z;
	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray

	float3 axis = make_float3(1, 1, 1);
	float3 C = make_float3(cube.x, cube.y, cube.z);
	float3 lb = C - cube.params.cubeParameters.size / 2 * axis;
	float3 rt = C + cube.params.cubeParameters.size / 2 * axis;
	float t1 = (lb.x - ray.origin.x) * dirfrac.x;
	float t2 = (rt.x - ray.origin.x) * dirfrac.x;
	float t3 = (lb.y - ray.origin.y) * dirfrac.y;
	float t4 = (rt.y - ray.origin.y) * dirfrac.y;
	float t5 = (lb.z - ray.origin.z) * dirfrac.z;
	float t6 = (rt.z - ray.origin.z) * dirfrac.z;

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
	float bias = 1.00001;
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
		bool gotoL = true;
		bool gotoR = true;
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