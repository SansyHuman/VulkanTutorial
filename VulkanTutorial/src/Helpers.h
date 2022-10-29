#pragma once

#include "stdafx.h"

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamiliy;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> transferFamily;

	bool isComplete();
};

struct SwapChainSupportDetails
{
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

std::vector<char> readFile(const std::string& filename);
