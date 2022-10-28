#pragma once

#include <cstdint>
#include <optional>
#include <fstream>
#include <vulkan/vulkan.hpp>

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamiliy;
	std::optional<uint32_t> presentFamily;

	bool isComplete();
};

struct SwapChainSupportDetails
{
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

std::vector<char> readFile(const std::string& filename);
