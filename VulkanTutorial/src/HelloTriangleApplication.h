#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#ifndef VULKAN_HPP_NO_NODISCARD_WARNINGS
#define VULKAN_HPP_NO_NODISCARD_WARNINGS
#endif
#ifndef VULKAN_HPP_NO_TO_STRING
#define VULKAN_HPP_NO_TO_STRING
#endif

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>

#include "Helpers.h"



class HelloTriangleApplication
{
private:
	GLFWwindow* window;

	vk::raii::Context context;
	vk::raii::Instance instance;

	vk::raii::SurfaceKHR surface;

	vk::raii::PhysicalDevice physicalDevice;
	vk::raii::Device device;

	vk::raii::Queue graphicsQueue;
	vk::raii::Queue presentQueue;

	vk::raii::SwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

	vk::raii::RenderPass renderPass;
	vk::raii::PipelineLayout pipelineLayout;
	vk::raii::Pipeline graphicsPipeline;

	static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

	vk::raii::CommandPool commandPool;
	vk::raii::CommandBuffers commandBuffers;

	std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;

	uint32_t currentFrame = 0;

	bool framebufferResized = false;

	const std::string TITLE = "Vulkan";
	const uint32_t WIDTH = 800;
	const uint32_t HEIGHT = 600;
	const double FRAME_STAT_UPDATE_INTERVAL_MS = 250.0;
	vk::Format swapChainImageFormat;
	vk::Extent2D swapChainExtent;

	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

#ifdef _DEBUG
	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	vk::raii::DebugUtilsMessengerEXT debugMessenger;
	std::ofstream logFile;
#endif

public:
	HelloTriangleApplication();

	void run();

private:
	void drawFrame();
	void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

	void cleanupSwapChain();
	void recreateSwapChain();

private:
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanup();

	void createInstance();

	void createSurface();

	void pickPhysicalDevice();
	bool isDeviceSuitable(const vk::raii::PhysicalDevice& device);
	bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& device);
	SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device);
	int rateDeviceSuitability(const vk::raii::PhysicalDevice& device);
	QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device);

	void createLogicalDevice();

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
	void createSwapChain();

	void createImageViews();

	void createRenderPass();

	void createGraphicsPipeline();
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);

	void createFramebuffers();

	void createCommandPool();
	void createCommandBuffer();

	void createSyncObjects();

	std::vector<const char*> getRequiredExtensions();

	bool checkValidationLayerSupport();
	void enumerateVkExtensions();
	vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo();
	void setupDebugMessenger();

	void calculateFrameStats();
};