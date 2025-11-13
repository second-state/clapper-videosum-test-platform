package summaryVideo

import (
	"context"
	"encoding/base64"
	"fmt"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"google.golang.org/genai"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// 读取图片并转换为base64 data URL
func readImageAsBase64(imagePath string) (string, error) {
	// 读取图片文件
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("读取图片失败: %w", err)
	}

	// 获取文件扩展名来确定MIME类型
	ext := strings.ToLower(filepath.Ext(imagePath))
	var mimeType string
	switch ext {
	case ".jpg", ".jpeg":
		mimeType = "image/jpeg"
	case ".png":
		mimeType = "image/png"
	case ".gif":
		mimeType = "image/gif"
	case ".webp":
		mimeType = "image/webp"
	default:
		return "", fmt.Errorf("不支持的图片格式: %s", ext)
	}

	// 转换为base64
	base64Str := base64.StdEncoding.EncodeToString(imageData)

	// 构造data URL
	dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, base64Str)

	return dataURL, nil
}

// 获取文件夹中的所有图片文件
func getImageFiles(folderPath string) ([]string, error) {
	var imageFiles []string

	// 支持的图片扩展名
	supportedExts := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".gif":  true,
		".webp": true,
	}

	err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// 跳过目录
		if info.IsDir() {
			return nil
		}

		// 检查是否是支持的图片格式
		ext := strings.ToLower(filepath.Ext(path))
		if supportedExts[ext] {
			imageFiles = append(imageFiles, path)
		}

		return nil
	})

	return imageFiles, err
}

// ProcessVideoFrames 主函数：遍历文件夹并发送带图片的请求
func ProcessVideoFrames(folderPath string, srtText string, description string, useOpenRouter bool, baseUrlArg, apiKeyArg, modelNameArg, promptTextArg string) (string, string, error) {
	var baseUrl, apiKey, modelName, promptText string

	pick := func(arg, envVal string) string {
		if arg != "" {
			return arg
		}
		return envVal
	}

	if useOpenRouter {
		baseUrl = pick(baseUrlArg, os.Getenv("OPENROUTER_BASE_URL"))
		apiKey = pick(apiKeyArg, os.Getenv("OPENROUTER_API_KEY"))
		modelName = pick(modelNameArg, os.Getenv("OPENROUTER_MODEL_NAME"))
		promptText = pick(promptTextArg, os.Getenv("OPENROUTER_PROMPT_TEXT"))
	} else {
		baseUrl = pick(baseUrlArg, os.Getenv("VLLM_BASE_URL"))
		apiKey = pick(apiKeyArg, os.Getenv("VLLM_API_KEY"))
		modelName = pick(modelNameArg, os.Getenv("VLLM_MODEL_NAME"))
		promptText = pick(promptTextArg, os.Getenv("VLLM_PROMPT_TEXT"))
	}

	// 创建OpenAI客户端
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseUrl), // 会自动查找环境变量
	)

	// 获取文件夹中的所有图片
	imagePaths, err := getImageFiles(folderPath)
	if err != nil {
		return "", modelName, fmt.Errorf("获取图片文件失败: %v", err)
	}

	if len(imagePaths) == 0 {
		return "", modelName, fmt.Errorf("文件夹中没有找到图片文件")
	}

	fmt.Printf("找到 %d 张图片\n", len(imagePaths))

	if srtText == "" {
		srtText = "**No subtitles available**"
	}
	if description == "" {
		description = "**No description provided**"
	}

	promptText = fmt.Sprintf(promptText, srtText, description)

	// 构建消息内容的切片
	var contentParts []openai.ChatCompletionContentPartUnionParam

	// 首先添加文本部分 - 使用 TextContentPart 辅助函数
	contentParts = append(contentParts, openai.TextContentPart(promptText))

	// 遍历图片并添加到消息中
	successCount := 0
	for i, imagePath := range imagePaths {
		fmt.Printf("处理图片 %d/%d: %s\n", i+1, len(imagePaths), imagePath)

		imageURL, err := readImageAsBase64(imagePath)
		if err != nil {
			fmt.Printf("无法处理图片 %s: %v\n", imagePath, err)
			continue
		}

		// 使用 ImageContentPart 辅助函数添加图片
		contentParts = append(contentParts, openai.ImageContentPart(
			openai.ChatCompletionContentPartImageImageURLParam{
				URL: imageURL,
			},
		))
		successCount++
	}

	if successCount == 0 {
		return "", modelName, fmt.Errorf("没有成功处理任何图片")
	}

	fmt.Printf("成功处理 %d 张图片，开始调用API...\n", successCount)

	// 创建聊天完成请求
	chatCompletion, err := client.Chat.Completions.New(
		context.TODO(),
		openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(contentParts),
			},
			Model:     modelName,
			MaxTokens: openai.Int(8192), // 设置较大的token限制
		},
	)

	if err != nil {
		return "", modelName, fmt.Errorf("API调用错误: %v", err)
	}

	// 输出结果：尝试从返回文本中提取 JSON（去掉代码块标记、语言标签，截取首个 JSON 对象）
	if len(chatCompletion.Choices) > 0 {
		content := strings.TrimSpace(chatCompletion.Choices[0].Message.Content)

		// 移除可能存在的代码块标记 ``` 和语言标签（如 ```json）
		if strings.HasPrefix(content, "```") {
			content = strings.ReplaceAll(content, "```", "")
			content = strings.TrimSpace(content)
			// 若以 json 开头，去掉首行
			if len(content) >= 4 && strings.ToLower(content[:4]) == "json" {
				if idx := strings.Index(content, "\n"); idx != -1 {
					content = strings.TrimSpace(content[idx+1:])
				}
			}
		}

		// 截取首个 JSON 对象（从第一个 '{' 到最后一个 '}'）
		start := strings.Index(content, "{")
		end := strings.LastIndex(content, "}")
		if start != -1 && end != -1 && end > start {
			jsonStr := strings.TrimSpace(content[start : end+1])
			return jsonStr, modelName, nil
		}

		// 回退：直接返回原始内容
		return content, modelName, nil
	} else {
		return "", modelName, fmt.Errorf("API没有返回任何内容")
	}
}

// ProcessVideoFramesWithAI 根据环境变量选择使用OpenAI或Gemini处理视频帧
func ProcessVideoFramesWithAI(folderPath string, srtText string, description string, useAltModel bool, APIUrl string, APIKey string, Model string, Prompt string) (string, string, error) {
	return ProcessVideoFrames(folderPath, srtText, description, useAltModel, APIUrl, APIKey, Model, Prompt)
}

// processWithGemini 使用Gemini API处理视频帧
func processWithGemini(folderPath string, srtText string, description string) (string, string, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	modelName := os.Getenv("GEMINI_MODEL_NAME")
	promptText := os.Getenv("PROMPT_TEXT")

	if apiKey == "" {
		return "", modelName, fmt.Errorf("未设置Gemini API密钥(GEMINI_API_KEY)")
	}
	if modelName == "" {
		modelName = "gemini-2.5-flash"
	}

	imagePaths, err := getImageFiles(folderPath)
	if err != nil {
		return "", modelName, fmt.Errorf("获取图片文件失败: %v", err)
	}
	if len(imagePaths) == 0 {
		return "", modelName, fmt.Errorf("文件夹中没有找到图片文件")
	}
	fmt.Printf("找到 %d 张图片\n", len(imagePaths))

	if srtText == "" {
		srtText = "**No subtitles available**"
	}
	if description == "" {
		description = "**No description provided**"
	}

	promptText = fmt.Sprintf(promptText, srtText, description)

	// 设置环境变量，genai.NewClient会自动读取GEMINI_API_KEY
	os.Setenv("GEMINI_API_KEY", apiKey)
	ctx := context.Background()
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		return "", modelName, fmt.Errorf("创建Gemini客户端失败: %v", err)
	}

	var parts []*genai.Part
	// 先加文本
	parts = append(parts, genai.NewPartFromText(promptText))

	successCount := 0
	for i, imagePath := range imagePaths {
		fmt.Printf("处理图片 %d/%d: %s\n", i+1, len(imagePaths), imagePath)
		imageData, err := os.ReadFile(imagePath)
		if err != nil {
			fmt.Printf("无法读取图片 %s: %v\n", imagePath, err)
			continue
		}
		ext := strings.ToLower(filepath.Ext(imagePath))
		var mimeType string
		switch ext {
		case ".jpg", ".jpeg":
			mimeType = "image/jpeg"
		case ".png":
			mimeType = "image/png"
		case ".gif":
			mimeType = "image/gif"
		case ".webp":
			mimeType = "image/webp"
		default:
			fmt.Printf("不支持的图片格式: %s\n", ext)
			continue
		}
		parts = append(parts, genai.NewPartFromBytes(imageData, mimeType))
		successCount++
	}
	if successCount == 0 {
		return "", modelName, fmt.Errorf("没有成功处理任何图片")
	}
	fmt.Printf("成功处理 %d 张图片，开始调用Gemini API...\n", successCount)

	contents := []*genai.Content{
		genai.NewContentFromParts(parts, genai.RoleUser),
	}

	result, err := client.Models.GenerateContent(
		ctx,
		modelName,
		contents,
		&genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"video_summary": {
						Type:        genai.TypeString,
						Description: "A single sentence summary of the video's core content and theme.",
					},
					"scenes": {
						Type:        genai.TypeString,
						Description: "Description of the environment, time, and location.",
					},
					"characters": {
						Type:        genai.TypeString,
						Description: "Description of main characters, including role, age, gender, and emotion.",
					},
					"actions": {
						Type:        genai.TypeString,
						Description: "Sequential description of the characters' actions and interactions.",
					},
					"environmental_details": {
						Type:        genai.TypeString,
						Description: "Description of important objects, background text, and logos/signs.",
					},
					"perspective_or_emotional": {
						Type:        genai.TypeString,
						Description: "Summary of the main viewpoint, theme, or overall emotional tone.",
					},
					"Illegal": {
						Type:        genai.TypeString,
						Description: "Violation type tag for prohibited content (e.g., 'Blood_Violence', 'Sexual_Content') or an empty string ('') if safe. Strictly no detailed description of the content.",
					},
					"logo": {
						Type:        genai.TypeString,
						Description: "The name of the video platform logo if present (e.g., 'YouTube'), otherwise an empty string ('').",
					},
					"isAIGenerated": {
						Type:        genai.TypeBoolean,
						Description: "A boolean: true if content is likely AI-generated, false otherwise.",
					},
				},
				PropertyOrdering: []string{
					"video_summary",
					"scenes",
					"characters",
					"actions",
					"environmental_details",
					"perspective_or_emotional",
					"Illegal",
					"logo",
					"isAIGenerated",
				},
			},
		},
	)
	if err != nil {
		// 如果是限流或 429 错误，降级回 Keye/OpenAI 继续尝试
		lowerErr := strings.ToLower(err.Error())
		if strings.Contains(lowerErr, "429") || strings.Contains(lowerErr, "rate limit") || strings.Contains(lowerErr, "quota") {
			log.Printf("Gemini 返回限流或 429，回退到 OpenRouter: %v\n", err)
			//return ProcessVideoFrames(folderPath, srtText, description, true)
		}
		log.Printf("Gemini 请求报错，回退到 Keye\n")
		//return ProcessVideoFrames(folderPath, srtText, description, false)
	}

	// 若未报错但返回内容为空，视为无结果，回退到 Keye/OpenAI
	if result == nil || strings.TrimSpace(result.Text()) == "" {
		log.Printf("Gemini 未返回可用内容，回退到 Keye\n")
		//return ProcessVideoFrames(folderPath, srtText, description, false)
	}
	return result.Text(), modelName, nil
}

func main() {
	// 示例1：处理整个文件夹
	//folderPath := "./keyframes"
	//srtText := ""
	//description := ""
	//result, model, err := ProcessVideoFramesWithAI(folderPath, srtText, description, false)
	//if err != nil {
	//	log.Printf("处理视频帧出错: %v\n", err)
	//	return
	//}
	//fmt.Println("处理结果:", result, "使用模型:", model)
}
