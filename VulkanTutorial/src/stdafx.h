#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define VULKAN_HPP_NO_NODISCARD_WARNINGS
#define VULKAN_HPP_NO_TO_STRING
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <optional>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <unordered_set>
#include <iomanip>
#include <sstream>