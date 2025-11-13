package speech

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"io"
	"mime/multipart"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// TranscribeVideoToText：从视频中提取 wav，然后向 OpenRouter 发送 multipart/form-data 请求进行转写。
// 返回转写文本或错误。依赖本地 ffmpeg 可用。
func TranscribeVideoToText(videoPath, outputDir, apiUrl, apiKey, model string) (string, error) {
	// 生成 wav 路径
	base := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
	wavPath := filepath.Join(outputDir, base+".wav")

	// 使用 ffmpeg 提取音频为 16k 单声道 PCM WAV
	cmd := exec.Command("ffmpeg", "-y", "-i", videoPath, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wavPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("ffmpeg 提取失败: %v, output: %s", err, string(out))
	}

	// 打开 wav 文件
	f, err := os.Open(wavPath)
	if err != nil {
		return "", fmt.Errorf("打开 wav 文件失败: %w", err)
	}
	defer f.Close()

	// 构建 multipart/form-data
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filepath.Base(wavPath))
	if err != nil {
		return "", fmt.Errorf("创建 form file 失败: %w", err)
	}
	if _, err := io.Copy(part, f); err != nil {
		return "", fmt.Errorf("写入文件到 form 失败: %w", err)
	}

	// model 字段
	_ = writer.WriteField("model", model)
	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("关闭 writer 失败: %w", err)
	}

	// 使用 openai SDK 进行转写
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(apiUrl), // 会自动查找环境变量
	)
	// 构建消息内容的切片
	var contentParts []openai.ChatCompletionContentPartUnionParam

	promptText := "Please transcribe the speech content from the following video recording audio and output it in a subtitle-style format.\nI require **only the pure spoken words** (dialogue), excluding all background music, sound effects, or other non-speech information.\n**Return only the subtitle content.** Do not include any polite phrases, introductions, explanations, or any other extra text."

	contentParts = append(contentParts, openai.TextContentPart(promptText))

	audioData, err := os.ReadFile(wavPath)
	if err != nil {
		panic(err)
	}
	base64AudioString := base64.StdEncoding.EncodeToString(audioData)
	contentParts = append(contentParts, openai.InputAudioContentPart(
		openai.ChatCompletionContentPartInputAudioInputAudioParam{
			Format: "wav",
			Data:   base64AudioString,
		},
	))

	chatCompletion, err := client.Chat.Completions.New(
		context.TODO(),
		openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(contentParts),
			},
			Model:     model,
			MaxTokens: openai.Int(8192), // 设置较大的token限制
		},
	)

	if err != nil {
		return "", fmt.Errorf("API request failed: %v", err)
	}

	if chatCompletion == nil || len(chatCompletion.Choices) == 0 {
		return "", fmt.Errorf("empty response from API: %+v", chatCompletion)
	}

	transcription := strings.TrimSpace(chatCompletion.Choices[0].Message.Content)
	return transcription, nil
}
