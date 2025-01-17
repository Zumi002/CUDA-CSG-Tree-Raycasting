#include "CSGTree.cuh"



CSGTree CSGTree::Parse(const std::string& text)
{

	CSGTree tree;
	std::vector<std::string> splited = split(text);

	//pair first = nodeIdx, second = childrenCount
	std::stack<std::pair<int, int>> nodesStack;

	int primitivesCount = 0;
	int nodesCount = 0;

	for (int i = 0; i < splited.size(); i++)
	{
		tree.nodes.push_back(CSGNode(-1, -1, -1, -1, -1));
		if (nodesCount!=0)
		{
			if (nodesStack.empty())
			{
				throw std::invalid_argument("Cannot parse");
			}

			auto* nodeInfo = &nodesStack.top();
			if (nodeInfo->second == 0)
			{
				tree.nodes[nodeInfo->first].left = nodesCount;
				tree.nodes[nodesCount].parent = nodeInfo->first;
				nodeInfo->second++;
			}
			else
			{
				tree.nodes[nodeInfo->first].right = nodesCount;
				tree.nodes[nodesCount].parent = nodeInfo->first;
				nodesStack.pop();
			}
		}

		if (splited[i] == "Union")
		{
			tree.nodes[nodesCount].type = CSGTree::NodeType::Union;
			nodesStack.push({ nodesCount, 0 });
		}
		else if (splited[i] == "Difference")
		{
			tree.nodes[nodesCount].type = CSGTree::NodeType::Difference;
			nodesStack.push({ nodesCount, 0 });
		}
		else if (splited[i] == "Intersection")
		{
			tree.nodes[nodesCount].type = CSGTree::NodeType::Intersection;
			nodesStack.push({ nodesCount, 0 });
		}
		else if (splited[i] == "Sphere")
		{
			tree.nodes[nodesCount].type = CSGTree::NodeType::Sphere;
			tree.nodes[nodesCount].primitiveIdx = primitivesCount;

			float x = std::stof(splited[i + 1]);
			float y = std::stof(splited[i + 2]);
			float z = std::stof(splited[i + 3]);

			if (splited[i + 4].size() != 6)
			{
				throw std::invalid_argument("Cannot parse color " + splited[i + 4]);
			}
			float r = color(splited[i + 4].substr(0, 2));
			float g = color(splited[i + 4].substr(2, 2));
			float b = color(splited[i + 4].substr(4, 2));

			float radius = std::stof(splited[i + 5]);

			tree.primitives.addSphere(primitivesCount, x, y, z, r, g, b, radius);
			i += 5;
			primitivesCount++;
		}
		else if (splited[i] == "Cylinder")
		{
			tree.nodes[nodesCount].type = CSGTree::NodeType::Cylinder;
			tree.nodes[nodesCount].primitiveIdx = primitivesCount;

			float x = std::stof(splited[i + 1]);
			float y = std::stof(splited[i + 2]);
			float z = std::stof(splited[i + 3]);

			if (splited[i + 4].size() != 6)
			{
				throw std::invalid_argument("Cannot parse color " + splited[i + 4]);
			}
			float r = color(splited[i + 4].substr(0, 2));
			float g = color(splited[i + 4].substr(2, 2));
			float b = color(splited[i + 4].substr(4, 2));

			float radius = std::stof(splited[i + 5]);
			float height = std::stof(splited[i + 6]);
			double rotX = std::stod(splited[i + 7]);
			double rotY = std::stod(splited[i + 8]);
			double rotZ = std::stod(splited[i + 9]);

			if (rotX > 360 || rotX < 0)
				throw std::invalid_argument("Invalid roation rotX should be in range [0, 360] deg");
			if (rotY > 360 || rotY < 0)
				throw std::invalid_argument("Invalid roation rotY should be in range [0, 360] deg");
			if (rotZ > 360 || rotZ < 0)
				throw std::invalid_argument("Invalid roation rotZ should be in range [0, 360] deg");


			tree.primitives.addCylinder(primitivesCount, x, y, z, r, g, b, radius, height, rotX, rotY, rotZ);
			i += 9;
			primitivesCount++;
		}
		else if (splited[i] == "Cube")
		{
			tree.nodes[nodesCount].type = CSGTree::NodeType::Cube;
			tree.nodes[nodesCount].primitiveIdx = primitivesCount;

			float x = std::stof(splited[i + 1]);
			float y = std::stof(splited[i + 2]);
			float z = std::stof(splited[i + 3]);

			if (splited[i + 4].size() != 6)
			{
				throw std::invalid_argument("Cannot parse color " + splited[i + 4]);
			}
			float r = color(splited[i + 4].substr(0, 2));
			float g = color(splited[i + 4].substr(2, 2));
			float b = color(splited[i + 4].substr(4, 2));

			float size = std::stof(splited[i + 5]);

			tree.primitives.addCube(primitivesCount, x, y, z, r, g, b, size);
			i += 5;
			primitivesCount++;
		}
		else
		{
			throw std::invalid_argument("Cannot parse - Unrecognized keyword: " + splited[i]);
		}

		nodesCount++;
	}

	if (nodesCount != 2*primitivesCount - 1)
		throw std::invalid_argument("Cannot parse - number of primitives do not match number of nodes");

	return tree;
}

std::vector<std::string> split(const std::string& text)
{
	std::vector<std::string> splitString;
	std::stringstream ss(text);

	std::string str;

	while (ss >> str)
	{
		splitString.push_back(str);
	}

	return splitString;
}

float color(const std::string& hex)
{
	int value = std::stoi(hex, nullptr, 16);
	return (float)value / 255;
}