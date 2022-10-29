#pragma once

#include "stdafx.h"

#include "Helpers.h"

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		vk::VertexInputBindingDescription bindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);

		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
		};

		return attributeDescriptions;
	}
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
	0, 1, 2, 0, 2, 3
};

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
	vk::raii::Queue transferQueue;

	vk::raii::SwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

	vk::raii::RenderPass renderPass;
	vk::raii::DescriptorSetLayout descriptorSetLayout;
	vk::raii::PipelineLayout pipelineLayout;
	vk::raii::Pipeline graphicsPipeline;

	static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

	vk::raii::CommandPool commandPool;
	vk::raii::CommandBuffers commandBuffers;
	vk::raii::CommandPool transferPool;
	vk::raii::CommandBuffer transferBuffer;

	vk::raii::Buffer vertexBuffer;
	vk::raii::DeviceMemory vertexBufferMemory;
	vk::raii::Buffer indexBuffer;
	vk::raii::DeviceMemory indexBufferMemory;
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;

	vk::raii::DescriptorPool descriptorPool;
	vk::raii::DescriptorSets descriptorSets;

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

	void updateUniformBuffer(uint32_t currentImage);
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
	void createDescriptorSetLayout();

	void createFramebuffers();

	void createCommandPool();

	void createVertexBuffer();
	void createIndexBuffer();
	void createUniformBuffer();
	void createDescriptorPool();
	void createDescriptorSets();
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
	void createBuffer(
		vk::DeviceSize size, vk::BufferUsageFlags usage,
		vk::SharingMode sharingMode,
		vk::ArrayProxyNoTemporaries<const uint32_t> const& queueFamilyIndices,
		vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);

	void createCommandBuffer();

	void createSyncObjects();

	std::vector<const char*> getRequiredExtensions();

	bool checkValidationLayerSupport();
	void enumerateVkExtensions();
	vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo();
	void setupDebugMessenger();

	void calculateFrameStats();
};