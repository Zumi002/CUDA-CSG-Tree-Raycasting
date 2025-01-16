#pragma once

struct SphereParameters
{
	float radius;
};

struct CylinderParameters
{
	float radius;
	float height;
};

union Parameters
{
	SphereParameters sphereParameters;
	CylinderParameters cylinderParameters;
};

struct Primitive
{
	int id;
	float x;
	float y;
	float z;

	float r;
	float g;
	float b;
	//If i add rotation
	//float* rotx;
	//float* roty;
	//float* rotz;


	//For more primitives use union
	Parameters params;

	Primitive(int Id, float X, float Y, float Z, float R, float G, float B, float Radius)
	{
		id = Id;
		x = X;
		y = Y;
		z = Z;
		r = R;
		g = G;
		b = B;
		params.sphereParameters.radius = Radius;
	}

};

struct Primitives
{
	std::vector<Primitive> primitives;

	void addSphere(int Id, float X, float Y, float Z, float R, float G, float B, float Radius)
	{
		primitives.push_back(Primitive(Id, X, Y, Z, R, G, B, Radius));
	}
};
