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