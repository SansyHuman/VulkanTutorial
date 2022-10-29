#include "stdafx.h"
#include "HelloTriangleApplication.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData
)
{
	std::ofstream* logFile = (std::ofstream*)pUserData;

	std::string severity;
	switch(messageSeverity)
	{
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
		severity = "Verbose";
		break;
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
		severity = "Info";
		break;
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
		severity = "Warning";
		break;
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
		severity = "Error";
		break;
	}

	std::string type;
	switch(messageType)
	{
	case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
		type = "General";
		break;
	case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
		type = "Validation";
		break;
	case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
		type = "Performance";
		break;
	case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
		type = "Device Address Binding";
		break;
	}

	auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	std::cerr << "Validation layer " << "[" << severity << "] "
		<< "(" << type << ") " << std::chrono::system_clock::now() << ": "
		<< pCallbackData->pMessage << std::endl;

	if(logFile->is_open())
	{
		(*logFile) << "Validation layer " << "[" << severity << "] "
			<< "(" << type << ") " << std::chrono::system_clock::now() << ": "
			<< pCallbackData->pMessage << std::endl;
	}

	return VK_FALSE;
}

HelloTriangleApplication::HelloTriangleApplication()
	: window(nullptr), context(), instance(nullptr), surface(nullptr),
	physicalDevice(nullptr), device(nullptr), graphicsQueue(nullptr),
	presentQueue(nullptr), transferQueue(nullptr), swapChain(nullptr),
	renderPass(nullptr), descriptorSetLayout(nullptr), pipelineLayout(nullptr),
	graphicsPipeline(nullptr), commandPool(nullptr), commandBuffers(nullptr),
	transferPool(nullptr), transferBuffer(nullptr), vertexBuffer(nullptr),
	vertexBufferMemory(nullptr), indexBuffer(nullptr), indexBufferMemory(nullptr),
	descriptorPool(nullptr), descriptorSets(nullptr)
#ifdef _DEBUG
	, debugMessenger(nullptr)
#endif
{

}

void HelloTriangleApplication::run()
{
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}

void HelloTriangleApplication::drawFrame()
{
	device.waitForFences(*inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

	std::pair<vk::Result, uint32_t> acquireResult;
	try
	{
		 acquireResult = swapChain.acquireNextImage(
			std::numeric_limits<uint64_t>::max(), *imageAvailableSemaphores[currentFrame]
		);
	}
	catch(const vk::OutOfDateKHRError&)
	{
		recreateSwapChain();
		return;
	}

	if(acquireResult.first != vk::Result::eSuccess && acquireResult.first != vk::Result::eSuboptimalKHR)
		throw std::runtime_error("Failed to acquire swap chain image.");

	uint32_t imageIndex = acquireResult.second;

	device.resetFences(*inFlightFences[currentFrame]);

	commandBuffers[currentFrame].reset();
	recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

	updateUniformBuffer(currentFrame);

	vk::SubmitInfo submitInfo;

	vk::Semaphore waitSemaphores[] = { *imageAvailableSemaphores[currentFrame]};
	vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	submitInfo.setWaitSemaphores(waitSemaphores)
		.setWaitDstStageMask(waitStages)
		.setCommandBuffers(*commandBuffers[currentFrame]);
	vk::Semaphore signalSemaphores[] = { *renderFinishedSemaphores[currentFrame]};
	submitInfo.setSignalSemaphores(signalSemaphores);

	graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

	vk::SwapchainKHR swapChains[] = { *swapChain };
	vk::PresentInfoKHR presentInfo;
	presentInfo.setWaitSemaphores(signalSemaphores)
		.setSwapchains(swapChains)
		.setImageIndices(imageIndex);
	vk::Result presentResult;

	try
	{
		presentResult = presentQueue.presentKHR(presentInfo);
	}
	catch(const vk::OutOfDateKHRError&)
	{
		goto RecreateAfterPresent;
	}

	if(presentResult == vk::Result::eSuboptimalKHR || framebufferResized)
	{
		RecreateAfterPresent:
		framebufferResized = false;
		recreateSwapChain();
	}
	else if(presentResult != vk::Result::eSuccess)
		throw std::runtime_error("Failed to present swap chain image.");

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void HelloTriangleApplication::updateUniformBuffer(uint32_t currentImage)
{
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();

	auto ms = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime);
	float time = (float)ms.count() * 1e-6f;

	UniformBufferObject ubo;
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), (float)swapChainExtent.width / swapChainExtent.height, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;

	void* data = uniformBuffersMemory[currentImage].mapMemory(0, sizeof(ubo));
	memcpy_s(data, sizeof(ubo), &ubo, sizeof(ubo));
	uniformBuffersMemory[currentImage].unmapMemory();
}

void HelloTriangleApplication::recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	vk::CommandBufferBeginInfo beginInfo;
	commandBuffer.begin(beginInfo);

	vk::ClearValue clearColor;
	clearColor.color.float32 = { { 0.0f, 0.0f, 0.0f, 1.0f } };
	vk::RenderPassBeginInfo renderPassInfo;
	renderPassInfo.setRenderPass(*renderPass)
		.setFramebuffer(*(swapChainFramebuffers[imageIndex]))
		.setRenderArea(vk::Rect2D({ 0, 0 }, swapChainExtent))
		.setClearValues(clearColor);
	commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

	vk::Buffer vertexBuffers[] = { *vertexBuffer };
	vk::DeviceSize offsets[] = { 0 };
	commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
	commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);

	vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
	commandBuffer.setViewport(0, viewport);

	vk::Rect2D scissor({ 0, 0 }, swapChainExtent);
	commandBuffer.setScissor(0, scissor);

	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);

	commandBuffer.drawIndexed((uint32_t)indices.size(), 1, 0, 0, 0);

	commandBuffer.endRenderPass();

	commandBuffer.end();
}

void HelloTriangleApplication::cleanupSwapChain()
{
	swapChainFramebuffers.clear();
	swapChainImageViews.clear();
	swapChain.clear();
}

void HelloTriangleApplication::recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while(width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	device.waitIdle();

	cleanupSwapChain();

	createSwapChain();
	createImageViews();
	createFramebuffers();
}

void HelloTriangleApplication::initWindow()
{
	if(glfwInit() == GLFW_FALSE)
		throw std::runtime_error("Failed to initialize GLFW.");

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	// glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(WIDTH, HEIGHT, TITLE.c_str(), nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);

	glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
		auto app = (HelloTriangleApplication*)glfwGetWindowUserPointer(window);
		app->framebufferResized = true;
	});
}

void HelloTriangleApplication::initVulkan()
{
	createInstance();
#ifdef _DEBUG
	setupDebugMessenger();
#endif
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createFramebuffers();
	createCommandPool();
	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffer();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffer();
	createSyncObjects();
}

void HelloTriangleApplication::mainLoop()
{
	while(!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		drawFrame();
		calculateFrameStats();
	}

	device.waitIdle();
}

void HelloTriangleApplication::cleanup()
{
	glfwDestroyWindow(window);
	glfwTerminate();

#ifdef _DEBUG
	if(logFile.is_open())
		logFile.close();
#endif
}

void HelloTriangleApplication::createInstance()
{
#ifdef _DEBUG
	if(!checkValidationLayerSupport())
		throw std::runtime_error("Validation layers requeste, but not available.");

	enumerateVkExtensions();
#endif

	vk::ApplicationInfo appInfo(
		"Hello Triangle",
		VK_MAKE_VERSION(1, 0, 0),
		"No Engine",
		VK_MAKE_VERSION(1, 0, 0),
		VK_API_VERSION_1_3
	);

#ifdef _DEBUG
	vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo = populateDebugMessengerCreateInfo();
#endif
	auto extensions = getRequiredExtensions();
	vk::InstanceCreateInfo createInfo;
	createInfo.setPApplicationInfo(&appInfo)
		.setEnabledExtensionCount((uint32_t)extensions.size())
		.setPpEnabledExtensionNames(extensions.data())
#ifdef _DEBUG
		.setEnabledLayerCount((uint32_t)validationLayers.size())
		.setPpEnabledLayerNames(validationLayers.data())
		.setPNext((void*)&debugCreateInfo);
#else
		.setEnabledLayerCount(0);
#endif
	instance = vk::raii::Instance(context, createInfo);
}

void HelloTriangleApplication::createSurface()
{
	VkSurfaceKHR nativeSurface = VK_NULL_HANDLE;
	auto result = glfwCreateWindowSurface(*instance, window, nullptr, &nativeSurface);
	if(result != VK_SUCCESS)
		throw std::runtime_error("Failed to create window surface.");

	surface = vk::raii::SurfaceKHR(instance, nativeSurface);
}

void HelloTriangleApplication::pickPhysicalDevice()
{
	const vk::raii::PhysicalDevices devices(instance);
	if(devices.size() == 0)
		throw std::runtime_error("Failed to find GPU with Vulkan support.");

	const vk::raii::PhysicalDevice* best = nullptr;
	int bestScore = 0;

	for(const auto& device : devices)
	{
		int score = rateDeviceSuitability(device);
		if(score > bestScore)
		{
			bestScore = score;
			best = &device;
		}
	}

	if(!best)
		throw std::runtime_error("Failed to find a suitable GPU.");

	physicalDevice = *best;
}

bool HelloTriangleApplication::isDeviceSuitable(const vk::raii::PhysicalDevice& device)
{
	auto properties = device.getProperties();
	auto features = device.getFeatures();
	auto indices = findQueueFamilies(device);

	bool extensionsSupported = checkDeviceExtensionSupport(device);
	bool swapChainAdequate = false;
	if(extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
		&& features.geometryShader && indices.isComplete()
		&& extensionsSupported && swapChainAdequate;
}

bool HelloTriangleApplication::checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& device)
{
	auto availableExtensions = device.enumerateDeviceExtensionProperties();

	std::unordered_set<std::string> availableSet;
	std::transform(
		availableExtensions.begin(),
		availableExtensions.end(),
		std::inserter(availableSet, availableSet.begin()),
		[](const vk::ExtensionProperties& extension) -> std::string {
			return extension.extensionName;
		}
	);

	for(const auto& extension : deviceExtensions)
	{
		if(!availableSet.contains(extension))
			return false;
	}

	return true;
}

SwapChainSupportDetails HelloTriangleApplication::querySwapChainSupport(const vk::raii::PhysicalDevice& device)
{
	SwapChainSupportDetails details;
	details.capabilities = device.getSurfaceCapabilitiesKHR(*surface);
	details.formats = device.getSurfaceFormatsKHR(*surface);
	details.presentModes = device.getSurfacePresentModesKHR(*surface);

	return details;
}

int HelloTriangleApplication::rateDeviceSuitability(const vk::raii::PhysicalDevice& device)
{
	auto properties = device.getProperties();
	auto features = device.getFeatures();
	auto indices = findQueueFamilies(device);

	bool extensionsSupported = checkDeviceExtensionSupport(device);
	bool swapChainAdequate = false;
	if(extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	if(!features.geometryShader
		|| !indices.isComplete()
		|| !extensionsSupported
		|| !swapChainAdequate)
	{
		return 0;
	}

	int score = 0;

	score += properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? 1000 : 0;
	score += properties.limits.maxImageDimension2D;
	
	return score;
}

QueueFamilyIndices HelloTriangleApplication::findQueueFamilies(const vk::raii::PhysicalDevice& device)
{
	QueueFamilyIndices indices;

	const auto queueFamilies = device.getQueueFamilyProperties();

	int i = 0;
	for(const auto& queueFamily : queueFamilies)
	{
		if(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			indices.graphicsFamiliy = i;
		if(device.getSurfaceSupportKHR(i, *surface))
			indices.presentFamily = i;
		if((queueFamily.queueFlags & vk::QueueFlagBits::eTransfer)
			&& !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics))
			indices.transferFamily = i;

		if(indices.isComplete())
			break;

		i++;
	}

	return indices;
}

void HelloTriangleApplication::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	float queuePriority = 1.0f;

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::unordered_set<uint32_t> uniqueQueueFamilies = { 
		indices.graphicsFamiliy.value(), indices.presentFamily.value(),
		indices.transferFamily.value()
	};

	for(uint32_t queueFamily : uniqueQueueFamilies)
	{
		vk::DeviceQueueCreateInfo queueCreateInfo;
		queueCreateInfo
			.setQueueFamilyIndex(queueFamily)
			.setQueuePriorities(queuePriority);

		queueCreateInfos.push_back(std::move(queueCreateInfo));
	}
	
	vk::PhysicalDeviceFeatures deviceFeatures;

	vk::DeviceCreateInfo createInfo;
	createInfo.setQueueCreateInfos(queueCreateInfos)
		.setPEnabledFeatures(&deviceFeatures)
		.setPEnabledExtensionNames(deviceExtensions)
#ifdef _DEBUG
		.setPEnabledLayerNames(validationLayers);
#else
		.setEnabledLayerCount(0);
#endif

	device = physicalDevice.createDevice(createInfo);
	graphicsQueue = device.getQueue(indices.graphicsFamiliy.value(), 0);
	presentQueue = device.getQueue(indices.presentFamily.value(), 0);
	transferQueue = device.getQueue(indices.transferFamily.value(), 0);
}

vk::SurfaceFormatKHR HelloTriangleApplication::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for(const auto& format : availableFormats)
	{
		if(format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear)
			return format;
	}

	return availableFormats[0];
}

vk::PresentModeKHR HelloTriangleApplication::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
	for(const auto& mode : availablePresentModes)
	{
		if(mode == vk::PresentModeKHR::eMailbox)
			return mode;
	}

	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D HelloTriangleApplication::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
	if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		return capabilities.currentExtent;

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	vk::Extent2D actualExtent(
		(uint32_t)width,
		(uint32_t)height
	);
	actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
	actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

	return actualExtent;
}

void HelloTriangleApplication::createSwapChain()
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

	vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		imageCount = swapChainSupport.capabilities.maxImageCount;

	vk::SwapchainCreateInfoKHR createInfo;
	createInfo.setSurface(*surface)
		.setMinImageCount(imageCount)
		.setImageFormat(surfaceFormat.format)
		.setImageColorSpace(surfaceFormat.colorSpace)
		.setImageExtent(extent)
		.setImageArrayLayers(1)
		.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphicsFamiliy.value(), indices.presentFamily.value() };

	if(indices.graphicsFamiliy != indices.presentFamily)
	{
		createInfo.setImageSharingMode(vk::SharingMode::eConcurrent)
			.setQueueFamilyIndices(queueFamilyIndices);
	}
	else
	{
		createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
	}
	createInfo.setPreTransform(swapChainSupport.capabilities.currentTransform)
		.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
		.setPresentMode(presentMode)
		.setClipped(VK_TRUE)
		.setOldSwapchain(VK_NULL_HANDLE);

	swapChain = device.createSwapchainKHR(createInfo);

	swapChainImages = swapChain.getImages();

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;
}

void HelloTriangleApplication::createImageViews()
{
	swapChainImageViews.clear();

	std::transform(
		swapChainImages.begin(),
		swapChainImages.end(),
		std::back_inserter(swapChainImageViews),
		[this](const vk::Image& image) -> vk::raii::ImageView {
			vk::ImageViewCreateInfo createInfo;
			createInfo.setImage(image)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(swapChainImageFormat)
				.setComponents(vk::ComponentMapping())
				.setSubresourceRange(vk::ImageSubresourceRange(
					vk::ImageAspectFlagBits::eColor,
					0, 1, 0, 1
				));

			return device.createImageView(createInfo);
		}
	);
}

void HelloTriangleApplication::createRenderPass()
{
	vk::AttachmentDescription colorAttachment;
	colorAttachment.setFormat(swapChainImageFormat)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		.setStoreOp(vk::AttachmentStoreOp::eStore)
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

	vk::SubpassDescription subpass;
	subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
		.setColorAttachments(colorAttachmentRef);

	vk::SubpassDependency dependency;
	dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL)
		.setDstSubpass(0)
		.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
		.setSrcAccessMask(vk::AccessFlagBits::eNone)
		.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
		.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

	vk::RenderPassCreateInfo renderPassInfo;
	renderPassInfo.setAttachments(colorAttachment)
		.setSubpasses(subpass)
		.setDependencies(dependency);

	renderPass = device.createRenderPass(renderPassInfo);
}

void HelloTriangleApplication::createGraphicsPipeline()
{
	auto vertShaderCode = readFile("./Shader/Basic.vert.spv");
	auto fragShaderCode = readFile("./Shader/Basic.frag.spv");

	auto vertShaderModule = createShaderModule(vertShaderCode);
	auto fragShaderModule = createShaderModule(fragShaderCode);

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
	vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex)
		.setModule(*vertShaderModule)
		.setPName("main");

	vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
	fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment)
		.setModule(*fragShaderModule)
		.setPName("main");

	vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	std::vector<vk::DynamicState> dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};

	vk::PipelineDynamicStateCreateInfo dynamicState;
	dynamicState.setDynamicStates(dynamicStates);

	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescription = Vertex::getAttributeDescriptions();
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
	vertexInputInfo.setVertexBindingDescriptions(bindingDescription)
		.setVertexAttributeDescriptions(attributeDescription);

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
	inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList)
		.setPrimitiveRestartEnable(VK_FALSE);

	vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
	vk::Rect2D scissor({ 0, 0 }, swapChainExtent);

	vk::PipelineViewportStateCreateInfo viewportState;
	viewportState.setViewports(viewport)
		.setScissors(scissor);

	vk::PipelineRasterizationStateCreateInfo rasterizer;
	rasterizer.setDepthClampEnable(VK_FALSE)
		.setRasterizerDiscardEnable(VK_FALSE)
		.setPolygonMode(vk::PolygonMode::eFill)
		.setLineWidth(1.0f)
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setFrontFace(vk::FrontFace::eCounterClockwise)
		.setDepthBiasEnable(VK_FALSE);

	vk::PipelineMultisampleStateCreateInfo multisampling;
	multisampling.setSampleShadingEnable(VK_FALSE)
		.setRasterizationSamples(vk::SampleCountFlagBits::e1);

	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
	colorBlendAttachment.setColorWriteMask(
		vk::ColorComponentFlagBits::eR |
		vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA
	).setBlendEnable(VK_FALSE);

	vk::PipelineColorBlendStateCreateInfo colorBlending;
	colorBlending.setLogicOpEnable(VK_FALSE)
		.setAttachments(colorBlendAttachment);

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
	pipelineLayoutInfo.setSetLayouts(*descriptorSetLayout);
	pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

	vk::GraphicsPipelineCreateInfo pipelineInfo;
	pipelineInfo.setStages(shaderStages)
		.setPVertexInputState(&vertexInputInfo)
		.setPInputAssemblyState(&inputAssembly)
		.setPViewportState(&viewportState)
		.setPRasterizationState(&rasterizer)
		.setPMultisampleState(&multisampling)
		.setPColorBlendState(&colorBlending)
		.setPDynamicState(&dynamicState)
		.setLayout(*pipelineLayout)
		.setRenderPass(*renderPass)
		.setSubpass(0)
		.setBasePipelineHandle(VK_NULL_HANDLE)
		.setBasePipelineIndex(-1);

	graphicsPipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo);
}

vk::raii::ShaderModule HelloTriangleApplication::createShaderModule(const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo;
	createInfo.setCodeSize(code.size())
		.setPCode((const uint32_t*)code.data());

	return device.createShaderModule(createInfo);
}

void HelloTriangleApplication::createDescriptorSetLayout()
{
	vk::DescriptorSetLayoutBinding uboLayoutBinding;
	uboLayoutBinding.setBinding(0)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(1)
		.setStageFlags(vk::ShaderStageFlagBits::eVertex);

	vk::DescriptorSetLayoutCreateInfo layoutInfo;
	layoutInfo.setBindings(uboLayoutBinding);

	descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
}

bool HelloTriangleApplication::checkValidationLayerSupport()
{
#ifdef _DEBUG
	auto availableLayers = context.enumerateInstanceLayerProperties();

	for(const char* layerName : validationLayers)
	{
		bool layerFound = false;
		for(const auto& layerProperties : availableLayers)
		{
			if(strcmp(layerName, layerProperties.layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if(!layerFound)
			return false;
	}
#endif

	return true;
}

void HelloTriangleApplication::createFramebuffers()
{
	swapChainFramebuffers.clear();

	std::transform(
		swapChainImageViews.begin(),
		swapChainImageViews.end(),
		std::back_inserter(swapChainFramebuffers),
		[this](const vk::raii::ImageView& view) -> vk::raii::Framebuffer {
			vk::ImageView attachments[] = { *view };
			vk::FramebufferCreateInfo framebufferInfo;
			framebufferInfo.setRenderPass(*renderPass)
				.setAttachments(attachments)
				.setWidth(swapChainExtent.width)
				.setHeight(swapChainExtent.height)
				.setLayers(1);

			return device.createFramebuffer(framebufferInfo);
		}
	);
}

void HelloTriangleApplication::createCommandPool()
{
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

	vk::CommandPoolCreateInfo poolInfo(
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		queueFamilyIndices.graphicsFamiliy.value()
	);

	commandPool = device.createCommandPool(poolInfo);

	vk::CommandPoolCreateInfo transferPoolInfo(
		vk::CommandPoolCreateFlagBits::eTransient,
		queueFamilyIndices.transferFamily.value()
	);

	transferPool = device.createCommandPool(transferPoolInfo);
}

void HelloTriangleApplication::createVertexBuffer()
{
	auto queueFamilyIndices = findQueueFamilies(physicalDevice);
	vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
	uint32_t sharingQueues[] = {
		queueFamilyIndices.graphicsFamiliy.value(),
		queueFamilyIndices.transferFamily.value()
	};

	vk::raii::Buffer stagingBuffer(nullptr);
	vk::raii::DeviceMemory stagingBufferMemory(nullptr);

	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::SharingMode::eExclusive,
		nullptr,
		vk::MemoryPropertyFlagBits::eHostVisible |
		vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory
	);

	void* data = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy_s(data, bufferSize, vertices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferDst |
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::SharingMode::eConcurrent,
		sharingQueues,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vertexBuffer,
		vertexBufferMemory
	);

	copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

	stagingBuffer.clear();
	stagingBufferMemory.clear();
}

void HelloTriangleApplication::createIndexBuffer()
{
	auto queueFamilyIndices = findQueueFamilies(physicalDevice);
	vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
	uint32_t sharingQueues[] = {
		queueFamilyIndices.graphicsFamiliy.value(),
		queueFamilyIndices.transferFamily.value()
	};

	vk::raii::Buffer stagingBuffer(nullptr);
	vk::raii::DeviceMemory stagingBufferMemory(nullptr);

	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::SharingMode::eExclusive,
		nullptr,
		vk::MemoryPropertyFlagBits::eHostVisible |
		vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory
	);

	void* data = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy_s(data, bufferSize, indices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferDst |
		vk::BufferUsageFlagBits::eIndexBuffer,
		vk::SharingMode::eConcurrent,
		sharingQueues,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		indexBuffer,
		indexBufferMemory
	);

	copyBuffer(stagingBuffer, indexBuffer, bufferSize);

	stagingBuffer.clear();
	stagingBufferMemory.clear();
}

void HelloTriangleApplication::createUniformBuffer()
{
	vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

	for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		uniformBuffers.push_back(nullptr);
		uniformBuffersMemory.push_back(nullptr);

		createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::SharingMode::eExclusive,
			nullptr,
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent,
			uniformBuffers[i],
			uniformBuffersMemory[i]
		);
	}
}

void HelloTriangleApplication::createDescriptorPool()
{
	vk::DescriptorPoolSize poolSize;
	poolSize.setType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(MAX_FRAMES_IN_FLIGHT);

	vk::DescriptorPoolCreateInfo poolInfo;
	poolInfo.setPoolSizes(poolSize)
		.setMaxSets(MAX_FRAMES_IN_FLIGHT);

	descriptorPool = device.createDescriptorPool(poolInfo);
}

void HelloTriangleApplication::createDescriptorSets()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);

	vk::DescriptorSetAllocateInfo allocInfo;
	allocInfo.setDescriptorPool(*descriptorPool)
		.setSetLayouts(layouts);

	descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

	for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo bufferInfo(*uniformBuffers[i], 0, sizeof(UniformBufferObject));

		vk::WriteDescriptorSet descriptorWrite;
		descriptorWrite.setDstSet(*descriptorSets[i])
			.setDstBinding(0)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setBufferInfo(bufferInfo);

		device.updateDescriptorSets(descriptorWrite, nullptr);
	}
}

uint32_t HelloTriangleApplication::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	auto memProperties = physicalDevice.getMemoryProperties();

	for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if(typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}

	throw std::runtime_error("Failed to find suitable memory type.");
}

void HelloTriangleApplication::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::SharingMode sharingMode, vk::ArrayProxyNoTemporaries<const uint32_t> const& queueFamilyIndices, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory)
{
	vk::BufferCreateInfo bufferInfo;
	bufferInfo.setSize(size)
		.setUsage(usage)
		.setSharingMode(sharingMode)
		.setQueueFamilyIndices(queueFamilyIndices);

	buffer = device.createBuffer(bufferInfo);

	auto memRequirements = buffer.getMemoryRequirements();

	vk::MemoryAllocateInfo allocInfo;
	allocInfo.setAllocationSize(memRequirements.size)
		.setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));

	bufferMemory = device.allocateMemory(allocInfo);

	buffer.bindMemory(*bufferMemory, 0);
}

void HelloTriangleApplication::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
{
	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.setLevel(vk::CommandBufferLevel::ePrimary)
		.setCommandPool(*transferPool)
		.setCommandBufferCount(1);

	vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(allocInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo;
	beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	commandBuffer.begin(beginInfo);

	vk::BufferCopy copyRegion(0, 0, size);
	commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

	commandBuffer.end();

	vk::SubmitInfo submitInfo;
	submitInfo.setCommandBuffers(*commandBuffer);

	transferQueue.submit(submitInfo);
	transferQueue.waitIdle();

	commandBuffer.clear();
}

void HelloTriangleApplication::createCommandBuffer()
{
	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.setCommandPool(*commandPool)
		.setLevel(vk::CommandBufferLevel::ePrimary)
		.setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);

	commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void HelloTriangleApplication::createSyncObjects()
{
	vk::SemaphoreCreateInfo semaphoreInfo;
	vk::FenceCreateInfo fenceInfo;
	fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);

	for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		imageAvailableSemaphores.push_back(device.createSemaphore(semaphoreInfo));
		renderFinishedSemaphores.push_back(device.createSemaphore(semaphoreInfo));
		inFlightFences.push_back(device.createFence(fenceInfo));
	}
}

void HelloTriangleApplication::enumerateVkExtensions()
{
	auto extensions = context.enumerateInstanceExtensionProperties();

	std::cout << "Available extensions:" << std::endl;

	for(const auto& extension : extensions)
		std::cout << "\t" << extension.extensionName << " Rev." << extension.specVersion << std::endl;
}

vk::DebugUtilsMessengerCreateInfoEXT HelloTriangleApplication::populateDebugMessengerCreateInfo()
{
#ifdef _DEBUG
	time_t rawtime;
	tm timeinfo;
	char buffer[80];

	time(&rawtime);
	localtime_s(&timeinfo, &rawtime);

	strftime(buffer, sizeof(buffer), "%d %m %Y %H %M %S", &timeinfo);
	std::string time(buffer);

	std::filesystem::create_directory("./Logs");
	
	std::string filename = "./Logs/Log " + time + ".txt";

	logFile = std::ofstream(filename);
#endif

	vk::DebugUtilsMessengerCreateInfoEXT createInfo;
	createInfo.setMessageSeverity(
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
	).setMessageType(
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
	).setPfnUserCallback(debugCallback);
#ifdef _DEBUG
	createInfo.setPUserData((void*)&logFile);
#endif

	return createInfo;
}

std::vector<const char*> HelloTriangleApplication::getRequiredExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef _DEBUG
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

	return extensions;
}

void HelloTriangleApplication::setupDebugMessenger()
{
#ifdef _DEBUG
	auto createInfo = populateDebugMessengerCreateInfo();

	debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
#endif
}

void HelloTriangleApplication::calculateFrameStats()
{
	static int cnt = 0;
	static double accum = 0.0;
	static auto prev = std::chrono::high_resolution_clock::now();

	auto now = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::microseconds>(now - prev);
	accum += (double)ms.count() / 1000.0;
	cnt++;

	if(accum >= FRAME_STAT_UPDATE_INTERVAL_MS)
	{
		double fps = (double)cnt / (accum / 1000.0);
		double mspf = accum / cnt;

		std::stringstream stream;
		stream << TITLE << " fps: " << std::fixed << std::setprecision(1) << fps
			<< "    mspf: " << std::setprecision(2) << mspf << " ms";
		glfwSetWindowTitle(window, stream.str().c_str());

		accum = 0;
		cnt = 0;
	}

	prev = now;
}
