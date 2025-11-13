package main

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"net/http"
	"os"
	"path/filepath"
	"videosum_test_platform/keyScore"
	"videosum_test_platform/saveVideo"
	"videosum_test_platform/speech"
	"videosum_test_platform/summaryVideo"
)

type Provider struct {
	APIUrl string `json:"api_url" binding:"required"`
	APIKey string `json:"api_key" binding:"required"`
	Model  string `json:"model" binding:"required"`
}

// 新增每个模型的 summary 列表项
type SummaryItem struct {
	TokensUsed      int    `json:"tokens_used"`
	DurationSeconds int    `json:"duration_seconds"`
	Summary         string `json:"summary"`
	Message         string `json:"message"`
	ModelName       string `json:"model_name"`
}

type VideoResult struct {
	VideoURL        string        `json:"video_url"`
	ImageList       []string      `json:"image_list"`
	TokensUsed      int           `json:"tokens_used"`
	DurationSeconds int           `json:"duration_seconds"`
	Summary         string        `json:"summary"`
	SummaryList     []SummaryItem `json:"summary_list"` // 新增字段
	Message         string        `json:"message"`
}

func main() {
	r := gin.Default()

	// 静态资源 (比如前端的 js/css)
	r.Static("/static", "./static")

	// 模板 (HTML 页面)
	r.LoadHTMLGlob("templates/*")

	// 页面路由
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", gin.H{
			"title": "视频对比测试平台",
		})
	})

	// API 路由
	r.POST("/api/test", func(c *gin.Context) {
		//{
		//	"video_url": "https://v12.myclapper.com/qvideos/nQP7OWZMPB/be981f5de16145c9aeb699029ad88b97_transcode_1375376.mp4",
		//	"prompt": "test",
		//	"providers": [
		//{
		//"api_url": "https://openrouter.ai/api/v1",
		//"api_key": "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",
		//"model": "google/gemini-2.5-flash"
		//}
		//]
		//}
		var req struct {
			VideoURL  string     `json:"video_url" binding:"required"`
			Providers []Provider `json:"providers" binding:"required"`
			Prompt    string     `json:"prompt" binding:"required"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// 这里调用你的视频处理逻辑
		// 比如直接调用本地函数或向 SQS 发消息
		result, err := processVideo(req.VideoURL, req.Providers, req.Prompt)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"message": err.Error()})
		} else {
			c.JSON(http.StatusOK, gin.H{
				"status": "success",
				"result": result,
			})
		}
	})

	r.Run(":8080")
}

func processVideo(videoURL string, providers []Provider, prompt string) ([]*VideoResult, error) {
	fmt.Println("视频链接:", videoURL)

	// 解析可能以 ';' 分隔的多个视频链接（并手动去除首尾空白以避免新增 import）
	parseAndTrim := func(s string) []string {
		var res []string
		cur := ""
		trim := func(t string) string {
			start, end := 0, len(t)-1
			for start <= end {
				c := t[start]
				if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
					start++
					continue
				}
				break
			}
			for end >= start {
				c := t[end]
				if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
					end--
					continue
				}
				break
			}
			if start > end {
				return ""
			}
			return t[start : end+1]
		}

		for i := 0; i < len(s); i++ {
			if s[i] == ';' {
				t := trim(cur)
				if t != "" {
					res = append(res, t)
				}
				cur = ""
			} else {
				cur += string(s[i])
			}
		}
		// 最后一个
		t := trim(cur)
		if t != "" {
			res = append(res, t)
		}
		return res
	}

	videoList := parseAndTrim(videoURL)
	if len(videoList) == 0 {
		return nil, fmt.Errorf("没有有效的视频链接")
	}

	var results []*VideoResult

	for _, v := range videoList {
		fmt.Printf("开始处理视频: %s\n", v)

		outputDir, file, err := saveVideo.DownloadVideo(v)
		if err != nil {
			fmt.Println("下载错误:", err)
			results = append(results, &VideoResult{
				VideoURL:        v,
				ImageList:       []string{},
				TokensUsed:      0,
				DurationSeconds: 0,
				Summary:         "",
				SummaryList:     []SummaryItem{},
				Message:         fmt.Sprintf("下载失败: %v", err),
			})
			continue
		}

		videoPath := filepath.Join(outputDir, file)
		fmt.Printf("下载完成！文件夹名: %s 视频文件名: %s\n", outputDir, file)

		var transcription string
		if len(providers) > 0 {
			apiURL := os.Getenv("WHISPER_API_URL")
			apiKey := os.Getenv("WHISPER_API_KEY")
			model := os.Getenv("WHISPER_API_MODEL")
			transcription, err = speech.TranscribeVideoToText(videoPath, outputDir, apiURL, apiKey, model)
			if err != nil {
				fmt.Println("语音转写失败:", err)
			} else {
				fmt.Println("语音转写结果:", transcription)
			}
		} else {
			fmt.Println("未提供 provider，跳过语音转写")
		}

		imageUrl, err := keyScore.ExtractKeyframes(videoPath, outputDir)
		if err != nil {
			fmt.Println("提取关键帧失败:", err)
			results = append(results, &VideoResult{
				VideoURL:        v,
				ImageList:       []string{},
				TokensUsed:      0,
				DurationSeconds: 0,
				Summary:         "",
				SummaryList:     []SummaryItem{},
				Message:         fmt.Sprintf("提取关键帧失败: %v", err),
			})
			continue
		}

		// 为每个 provider 生成一个 SummaryItem，然后将它们都放入 VideoResult.SummaryList
		var summaryList []SummaryItem
		firstSuccessSummary := ""
		for _, prov := range providers {
			summarized, useModel, summarizedErr := summaryVideo.ProcessVideoFramesWithAI(outputDir, transcription, "", false, prov.APIUrl, prov.APIKey, prov.Model, prompt)

			item := SummaryItem{
				TokensUsed:      0, // 如果 summaryVideo 返回此类信息，可在此填充
				DurationSeconds: 0, // 同上
				Summary:         "",
				Message:         "",
				ModelName:       prov.Model,
			}

			if summarizedErr != nil {
				item.Message = fmt.Sprintf("处理失败: %v", summarizedErr)
				item.Summary = ""
				fmt.Println("provider 出错:", prov.APIUrl, summarizedErr)
			} else {
				item.Message = "处理成功"
				// 如果 summaryVideo 返回空字符串但 useModel 或其他信息可用，也可以据此填充
				item.Summary = summarized
				// 第一个成功的 summary 保留到顶层 Summary 以兼容旧客户端
				if firstSuccessSummary == "" && summarized != "" {
					firstSuccessSummary = summarized
				}
				// 可以根据 useModel 或其他返回值决定 tokens/duration 的填充（目前无）
				_ = useModel
			}

			summaryList = append(summaryList, item)
			// 注意：不再遇到第一个成功就 break，而是收集所有 provider 的结果
		}

		overallMsg := "处理完成"
		if len(summaryList) == 0 {
			overallMsg = "没有可用的 provider 结果"
		}

		results = append(results, &VideoResult{
			VideoURL:        v,
			ImageList:       imageUrl,
			TokensUsed:      0,
			DurationSeconds: 0,
			Summary:         firstSuccessSummary,
			SummaryList:     summaryList,
			Message:         overallMsg,
		})
	}

	return results, nil
}
