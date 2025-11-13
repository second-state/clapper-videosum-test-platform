package saveVideo

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

var errNotS3URL = errors.New("not an s3 url")

const maxPixels = 100352.0 // 使用浮点数进行精确计算

// VideoInfo 结构用于解析 FFprobe 的输出
type VideoInfo struct {
	Streams []struct {
		CodecType string `json:"codec_type"`
		Width     int    `json:"width"`
		Height    int    `json:"height"`
	} `json:"streams"`
}

// getDimensions 使用 ffprobe 获取视频的宽度和高度
func getDimensions(inputPath string) (*VideoInfo, error) {
	cmd := exec.Command("ffprobe",
		"-v", "quiet",
		"-select_streams", "v",
		"-show_streams",
		"-of", "json",
		inputPath)

	output, err := cmd.Output()
	if err != nil {
		// 检查 ffprobe 是否在 PATH 中
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("ffprobe 执行错误: %s, 检查 ffprobe 是否已安装并添加到 PATH", string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("执行 ffprobe 失败: %w", err)
	}

	var info VideoInfo
	if err := json.Unmarshal(output, &info); err != nil {
		return nil, fmt.Errorf("解析 ffprobe JSON 输出失败: %w", err)
	}

	return &info, nil
}

func convertVideo(inputPath, outputPath string) error {
	// 获取视频元数据 (FFprobe)
	info, err := getDimensions(inputPath)
	if err != nil {
		return fmt.Errorf("无法获取视频信息: %w", err)
	}

	var originalW, originalH float64 // 使用 float64 进行精确计算
	for _, stream := range info.Streams {
		if stream.CodecType == "video" {
			originalW = float64(stream.Width)
			originalH = float64(stream.Height)
			break
		}
	}

	if originalW == 0 || originalH == 0 {
		return fmt.Errorf("视频流未找到或分辨率为零")
	}

	originalPixels := originalW * originalH

	// 如果原始像素数已小于或等于限制，则无需缩放（仅进行转码）
	if originalPixels <= maxPixels {
		fmt.Printf("视频像素 (%.0f) 小于限制 (%.0f)，无需缩放。\n", originalPixels, maxPixels)
		// 直接调用 FFmpeg 进行转码，保持原始分辨率
		// ffmpeg -i input.mp4 -r 30 -c:v libx264 -pix_fmt yuvj420p -r 30 -c:v libx264 -crf 18 -preset medium -color_range 2 -colorspace 1 -color_primaries 1 -color_trc 1 -an -y output.mp4
		cmdArgs := []string{
			"-i", inputPath,
			"-r", "30", // 设置输出帧率为30fps
			"-c:v", "libx264", // 使用H.264重新编码视频
			"-pix_fmt", "yuvj420p", // 像素格式
			"-r", "30", // 设置输出帧率为30fps
			"-c:v", "libx264", // 视频编码器
			"-crf", "18", // 质量参数
			"-preset", "medium", // 编码速度/效率预设
			"-color_range", "2", // 全范围色彩
			"-colorspace", "1", // BT.709 色彩空间
			"-color_primaries", "1", // BT.709 色彩基准
			"-color_trc", "1", // BT.709 色调响应曲线
			"-an", // 音频直接复制
			"-y",
			outputPath,
		}

		cmd := exec.Command("ffmpeg", cmdArgs...)
		fmt.Println("执行 FFmpeg (复制流) 命令:", cmd.Args)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("FFmpeg (复制流) 执行失败: %w", err)
		}
		return nil
	}

	// 目标宽高比
	aspectRatio := originalW / originalH

	// 计算最大宽度
	targetWFloat := math.Sqrt(maxPixels * aspectRatio)

	// 根据宽高比计算最大高度
	targetHFloat := targetWFloat / aspectRatio

	// 结果向下取整到最近的整数
	targetW := int(math.Floor(targetWFloat))
	targetH := int(math.Floor(targetHFloat))

	// 确保宽度和高度都是偶数 (向下取整到最近偶数)
	targetW = (targetW / 2) * 2
	targetH = (targetH / 2) * 2

	// 如果向下取整导致其中一个维度变成0，则再检查一次
	if targetW == 0 || targetH == 0 {
		// 这很少发生，但如果发生，使用原分辨率
		targetW = int(originalW)
		targetH = int(originalH)
		// 重新确保偶数
		targetW = (targetW / 2) * 2
		targetH = (targetH / 2) * 2
	}

	finalPixels := targetW * targetH
	if finalPixels > maxPixels {
		fmt.Printf("警告: 校验发现像素超标 (%d)，重新调整...\n", finalPixels)
		if targetW > targetH {
			targetW -= 2 // 降低宽度
		} else {
			targetH -= 2 // 降低高度
		}
		// 重新计算 finalPixels
		finalPixels = targetW * targetH
	}

	// 增加日志输出，确认计算结果
	fmt.Printf("原始尺寸: %.0fx%.0f, 目标缩放尺寸: %dx%d, 最终像素: %d\n",
		originalW, originalH, targetW, targetH, finalPixels)

	if finalPixels > 100352 {
		return fmt.Errorf("无法将像素限制在 100352 以下，当前为 %d", finalPixels)
	}

	// 3. 调用 FFmpeg 执行转码
	scaleParam := fmt.Sprintf("scale=%d:%d:flags=lanczos", targetW, targetH)

	// 构建 FFmpeg 命令
	// ffmpeg -i input.mp4 -vf "scale=238:420:flags=lanczos" -c:v libx264 -pix_fmt yuvj420p -r 30 -crf 18 -preset medium -color_range 2 -colorspace 1 -color_primaries 1 -color_trc 1 output.mp4
	cmdArgs := []string{
		"-i", inputPath,
		"-vf", scaleParam, // 视频滤镜：缩放
		"-pix_fmt", "yuvj420p", // 像素格式
		"-r", "30", // 设置输出帧率为30fps
		"-c:v", "libx264", // 视频编码器
		"-crf", "18", // 质量参数
		"-preset", "medium", // 编码速度/效率预设
		"-color_range", "2", // 全范围色彩
		"-colorspace", "1", // BT.709 色彩空间
		"-color_primaries", "1", // BT.709 色彩基准
		"-color_trc", "1", // BT.709 色调响应曲线
		"-an", // 不处理音频
		"-y",
		outputPath,
	}

	cmd := exec.Command("ffmpeg", cmdArgs...)

	fmt.Println("执行 FFmpeg 命令:", cmd.Args)

	// 设置输出和错误流，以便在出现问题时进行调试
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("FFmpeg 执行失败: %w", err)
	}

	return nil
}

// 生成随机字符串
func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	bytes := make([]byte, n)
	if _, err := rand.Read(bytes); err != nil {
		panic(err)
	}
	for i, b := range bytes {
		bytes[i] = letters[b%byte(len(letters))]
	}
	return string(bytes)
}

func DownloadVideo(vUrl string) (string, string, error) {
	// 生成随机文件夹名
	folderName := randomString(16)

	// 创建文件夹
	err := os.Mkdir(folderName, 0755)
	if err != nil {
		return "", "", fmt.Errorf("创建文件夹失败: %v", err)
	}

	// 设置清理函数，在发生错误时删除创建的文件夹
	cleanup := func(shouldCleanup bool) {
		if shouldCleanup {
			os.RemoveAll(folderName)
		}
	}

	// 从 URL 提取文件名
	fileName := filepath.Base(vUrl)
	if !strings.Contains(fileName, ".") { // 如果 URL 没有扩展名，就默认 mp4
		fileName += ".mp4"
	}

	// 目标文件路径
	filePath := filepath.Join(folderName, fileName)

	resp, err := http.Get(vUrl)
	if err != nil {
		cleanup(true)
		return "", "", fmt.Errorf("下载失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		cleanup(true)
		return "", "", fmt.Errorf("下载失败，状态码: %d", resp.StatusCode)
	}

	out, err := os.Create(filePath)
	if err != nil {
		cleanup(true)
		return "", "", fmt.Errorf("创建文件失败: %w", err)
	}
	defer out.Close()

	if _, err := io.Copy(out, resp.Body); err != nil {
		cleanup(true)
		return "", "", fmt.Errorf("保存文件失败: %w", err)
	}

	// 使用临时文件避免输入和输出为同一路径导致 FFmpeg 失败
	tmpOutput := filePath + ".tmp.mp4"

	fmt.Printf("开始转码: %s -> %s\n", filePath, tmpOutput)

	// 关闭并刷新之前创建的目标文件句柄
	if err := out.Sync(); err != nil {
		os.Remove(tmpOutput)
		cleanup(true)
		return "", "", fmt.Errorf("同步文件失败: %w", err)
	}
	if err := out.Close(); err != nil {
		os.Remove(tmpOutput)
		cleanup(true)
		return "", "", fmt.Errorf("关闭文件失败: %w", err)
	}

	if err := convertVideo(filePath, tmpOutput); err != nil {
		// 若转码失败，清理临时文件并返回错误
		os.Remove(tmpOutput)
		cleanup(true)
		return "", "", fmt.Errorf("转码失败: %w", err)
	}

	// 在 Windows 上直接重命名会失败（目标存在），先删除原文件再重命名
	if err := os.Remove(filePath); err != nil {
		os.Remove(tmpOutput)
		cleanup(true)
		return "", "", fmt.Errorf("删除原文件失败: %w", err)
	}
	if err := os.Rename(tmpOutput, filePath); err != nil {
		os.Remove(tmpOutput)
		cleanup(true)
		return "", "", fmt.Errorf("替换文件失败: %w", err)
	}

	return folderName, fileName, nil
}

func extractS3BucketAndKey(u *url.URL) (string, string, error) {
	if u == nil {
		return "", "", errNotS3URL
	}

	host := stripPort(u.Host)
	path := strings.TrimPrefix(u.Path, "/")

	if u.Scheme == "s3" {
		if host == "" || path == "" {
			return "", "", fmt.Errorf("无效的S3 URL: %s", u.String())
		}
		return host, path, nil
	}

	hostLower := strings.ToLower(host)
	if idx := strings.Index(hostLower, ".s3"); idx > 0 {
		bucket := host[:idx]
		if bucket == "" || path == "" {
			return "", "", fmt.Errorf("无效的S3 URL: %s", u.String())
		}
		return bucket, path, nil
	}

	if looksLikeS3Host(hostLower) {
		parts := strings.SplitN(path, "/", 2)
		if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
			return "", "", fmt.Errorf("无效的S3 URL格式: %s", u.String())
		}
		return parts[0], parts[1], nil
	}

	return "", "", errNotS3URL
}

func looksLikeS3Host(host string) bool {
	return strings.HasPrefix(host, "s3.") ||
		strings.HasPrefix(host, "s3-") ||
		strings.Contains(host, ".s3.") ||
		strings.HasSuffix(host, ".amazonaws.com") ||
		host == "s3"
}

func stripPort(host string) string {
	if idx := strings.Index(host, ":"); idx != -1 {
		return host[:idx]
	}
	return host
}

func main() {
	url := "https://example.com/video.mp4"

	folder, file, err := DownloadVideo(url)
	if err != nil {
		fmt.Println("错误:", err)
		return
	}

	fmt.Printf("下载完成！\n文件夹名: %s\n视频文件名: %s\n", folder, file)
}
