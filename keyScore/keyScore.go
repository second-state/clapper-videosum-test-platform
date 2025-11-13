package keyScore

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Frame 表示一个视频帧
type Frame struct {
	Index     int     // 帧索引
	Timestamp float64 // 时间戳（秒）
	Score     float64 // 总分
	IsIFrame  bool    // 是否为I帧
	Image     image.Image
	Hash      uint64 // 用于相似度比较的哈希值
}

// VideoInfo 存储视频信息
type VideoInfo struct {
	Duration    float64
	FrameRate   float64
	TotalFrames int
	Width       int
	Height      int
}

// Config 配置参数
type Config struct {
	VideoPath string
	OutputDir string
	TempDir   string
}

// 获取视频信息
func getVideoInfo(videoPath string) (*VideoInfo, error) {
	cmd := exec.Command("ffprobe", "-v", "error", "-show_entries",
		"format=duration:stream=width,height,r_frame_rate,nb_frames",
		"-of", "json", videoPath)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe error: %v", err)
	}

	var result struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
		Streams []struct {
			Width      int    `json:"width"`
			Height     int    `json:"height"`
			RFrameRate string `json:"r_frame_rate"`
			NbFrames   string `json:"nb_frames"`
		} `json:"streams"`
	}

	if err := json.Unmarshal(output, &result); err != nil {
		return nil, err
	}

	duration, _ := strconv.ParseFloat(result.Format.Duration, 64)

	// 解析帧率
	rateParts := strings.Split(result.Streams[0].RFrameRate, "/")
	frameRate := 30.0 // 默认值
	if len(rateParts) == 2 {
		num, _ := strconv.ParseFloat(rateParts[0], 64)
		den, _ := strconv.ParseFloat(rateParts[1], 64)
		if den != 0 {
			frameRate = num / den
		}
	}

	totalFrames, _ := strconv.Atoi(result.Streams[0].NbFrames)

	return &VideoInfo{
		Duration:    duration,
		FrameRate:   frameRate,
		TotalFrames: totalFrames,
		Width:       result.Streams[0].Width,
		Height:      result.Streams[0].Height,
	}, nil
}

// 获取I帧列表
func getIFrames(videoPath string) (map[int]bool, error) {
	cmd := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0",
		"-show_entries", "frame=pict_type", "-of", "csv=p=0", videoPath)

	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	iFrames := make(map[int]bool)
	lines := strings.Split(string(output), "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) == "I" {
			iFrames[i] = true
		}
	}

	return iFrames, nil
}

// 提取帧
func extractFrame(videoPath string, timestamp float64) (image.Image, error) {
	cmd := exec.Command("ffmpeg", "-ss", fmt.Sprintf("%.3f", timestamp),
		"-i", videoPath, "-frames:v", "1", "-f", "image2pipe",
		"-vcodec", "mjpeg", "-")

	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	img, err := jpeg.Decode(bytes.NewReader(output))
	if err != nil {
		return nil, err
	}

	return img, nil
}

// 计算RGB标准差
func calculateRGBStdDev(img image.Image) float64 {
	bounds := img.Bounds()
	var rSum, gSum, bSum float64
	var rSum2, gSum2, bSum2 float64
	pixelCount := float64(bounds.Dx() * bounds.Dy())

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rf, gf, bf := float64(r>>8), float64(g>>8), float64(b>>8)

			rSum += rf
			gSum += gf
			bSum += bf

			rSum2 += rf * rf
			gSum2 += gf * gf
			bSum2 += bf * bf
		}
	}

	rMean := rSum / pixelCount
	gMean := gSum / pixelCount
	bMean := bSum / pixelCount

	rVar := (rSum2 / pixelCount) - (rMean * rMean)
	gVar := (gSum2 / pixelCount) - (gMean * gMean)
	bVar := (bSum2 / pixelCount) - (bMean * bMean)

	return math.Sqrt((rVar + gVar + bVar) / 3)
}

// 转换为灰度图
func toGrayscale(img image.Image) *image.Gray {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			grayVal := uint8((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
			gray.Set(x, y, color.Gray{grayVal})
		}
	}

	return gray
}

// 简化的Canny边缘检测（使用Sobel算子）
func calculateEdgeRatio(gray *image.Gray) float64 {
	bounds := gray.Bounds()
	edges := 0
	total := 0

	// Sobel算子
	for y := bounds.Min.Y + 1; y < bounds.Max.Y-1; y++ {
		for x := bounds.Min.X + 1; x < bounds.Max.X-1; x++ {
			// Gx
			gx := -int(gray.GrayAt(x-1, y-1).Y) - 2*int(gray.GrayAt(x-1, y).Y) - int(gray.GrayAt(x-1, y+1).Y) +
				int(gray.GrayAt(x+1, y-1).Y) + 2*int(gray.GrayAt(x+1, y).Y) + int(gray.GrayAt(x+1, y+1).Y)

			// Gy
			gy := -int(gray.GrayAt(x-1, y-1).Y) - 2*int(gray.GrayAt(x, y-1).Y) - int(gray.GrayAt(x+1, y-1).Y) +
				int(gray.GrayAt(x-1, y+1).Y) + 2*int(gray.GrayAt(x, y+1).Y) + int(gray.GrayAt(x+1, y+1).Y)

			magnitude := math.Sqrt(float64(gx*gx + gy*gy))
			if magnitude > 50 { // 阈值
				edges++
			}
			total++
		}
	}

	return float64(edges) / float64(total)
}

// 计算拉普拉斯方差
func calculateLaplacianVariance(gray *image.Gray) float64 {
	bounds := gray.Bounds()
	var sum, sum2 float64
	count := 0

	// 拉普拉斯算子
	kernel := [][]int{
		{0, 1, 0},
		{1, -4, 1},
		{0, 1, 0},
	}

	for y := bounds.Min.Y + 1; y < bounds.Max.Y-1; y++ {
		for x := bounds.Min.X + 1; x < bounds.Max.X-1; x++ {
			var laplacian float64
			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					pixel := float64(gray.GrayAt(x+kx, y+ky).Y)
					laplacian += pixel * float64(kernel[ky+1][kx+1])
				}
			}

			sum += laplacian
			sum2 += laplacian * laplacian
			count++
		}
	}

	mean := sum / float64(count)
	variance := (sum2 / float64(count)) - (mean * mean)

	return variance
}

// 计算感知哈希（用于相似度比较）
func calculatePerceptualHash(img image.Image) uint64 {
	// 缩放到8x8
	small := image.NewGray(image.Rect(0, 0, 8, 8))
	bounds := img.Bounds()

	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			srcX := bounds.Min.X + (x * bounds.Dx() / 8)
			srcY := bounds.Min.Y + (y * bounds.Dy() / 8)
			r, g, b, _ := img.At(srcX, srcY).RGBA()
			gray := uint8((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
			small.Set(x, y, color.Gray{gray})
		}
	}

	// 计算平均值
	var sum uint32
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			sum += uint32(small.GrayAt(x, y).Y)
		}
	}
	avg := sum / 64

	// 生成哈希
	var hash uint64
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if uint32(small.GrayAt(x, y).Y) > avg {
				hash |= 1 << uint(y*8+x)
			}
		}
	}

	return hash
}

// 计算哈希差异
func hashDifference(h1, h2 uint64) int {
	diff := h1 ^ h2
	count := 0
	for diff != 0 {
		count++
		diff &= diff - 1
	}
	return count
}

// 评分函数
func scoreFrame(frame *Frame, rgbStdDev, edgeRatio, laplacianVar float64,
	allRGBStdDevs, allEdgeRatios, allLaplacianVars []float64) {

	score := 0.0

	// I帧加分
	if frame.IsIFrame {
		score += 20
	}

	// RGB标准差评分（0-20分）
	rgbScore := normalizeScore(rgbStdDev, allRGBStdDevs, 0, 20)
	score += rgbScore

	// 边缘比率评分（0-20分）
	edgeScore := normalizeScore(edgeRatio, allEdgeRatios, 0, 20)
	score += edgeScore

	// 拉普拉斯方差评分（0-40分）
	lapScore := normalizeScore(laplacianVar, allLaplacianVars, 0, 40)
	score += lapScore

	frame.Score = score
}

// 归一化评分
func normalizeScore(value float64, allValues []float64, minScore, maxScore float64) float64 {
	if len(allValues) == 0 {
		return minScore
	}

	sorted := make([]float64, len(allValues))
	copy(sorted, allValues)
	sort.Float64s(sorted)

	min := sorted[0]
	max := sorted[len(sorted)-1]

	if max == min {
		return (minScore + maxScore) / 2
	}

	normalized := (value - min) / (max - min)
	return minScore + normalized*(maxScore-minScore)
}

// 计算需要选择的帧数
func calculateFrameCount(duration float64) int {
	switch {
	case duration <= 60:
		return 5 + int((duration/60)*3) // 5-8帧
	case duration <= 120:
		return 8 + int(((duration-60)/60)*4) // 8-12帧
	case duration <= 180:
		return 12 + int(((duration-120)/60)*4) // 12-16帧
	default:
		return 16
	}
}

func SaveTempImageToS3(frames []*Frame, bizId string) ([]string, error) {

	links := make([]string, 0, len(frames))
	for i, frame := range frames {
		// 编码到内存
		var buf bytes.Buffer
		if frame.Image == nil {
			fmt.Printf("帧 %d 的 Image 为 nil，跳过\n", i)
			continue
		}
		b := frame.Image.Bounds()
		if b.Dx() <= 0 || b.Dy() <= 0 {
			fmt.Printf("帧 %d 的 Image 尺寸异常: %v，跳过\n", i, b)
			continue
		}
		// 确保图像为可编码的 RGBA，手动拷贝避免额外 import
		rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				rgba.Set(x-b.Min.X, y-b.Min.Y, frame.Image.At(x, y))
			}
		}
		if err := jpeg.Encode(&buf, rgba, &jpeg.Options{Quality: 95}); err != nil {
			fmt.Printf("编码帧 %d 到 JPEG 失败: %v (类型=%T, bounds=%v)\n", i, err, frame.Image, b)
			continue
		}

		// 将图片编码为 base64 并以 data URL 形式返回
		b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
		dataURL := "data:image/jpeg;base64," + b64
		links = append(links, dataURL)
	}

	return links, nil
}

// 选择关键帧
func selectKeyFrames(frames []*Frame, videoInfo *VideoInfo, targetCount int) []*Frame {
	// 按分数排序
	sort.Slice(frames, func(i, j int) bool {
		return frames[i].Score > frames[j].Score
	})

	// 计算各部分的帧数
	startFrames := int(math.Ceil(float64(targetCount) * 0.3))
	middleFrames := int(math.Ceil(float64(targetCount) * 0.5))
	endFrames := targetCount - startFrames - middleFrames

	// 定义各部分的时间范围
	startEnd := videoInfo.Duration * 0.3
	middleEnd := videoInfo.Duration * 0.7

	// 分组帧
	var startGroup, middleGroup, endGroup []*Frame
	for _, frame := range frames {
		if frame.Timestamp <= startEnd {
			startGroup = append(startGroup, frame)
		} else if frame.Timestamp <= middleEnd {
			middleGroup = append(middleGroup, frame)
		} else {
			endGroup = append(endGroup, frame)
		}
	}

	// 从每个组中选择帧
	selected := make([]*Frame, 0, targetCount)

	// 理想帧间距
	idealInterval := videoInfo.Duration / float64(targetCount)

	// 选择函数
	selectFromGroup := func(group []*Frame, count int, mustHaveFirst, mustHaveLast bool) []*Frame {
		if len(group) == 0 || count == 0 {
			return nil
		}

		result := make([]*Frame, 0, count)

		// 如果必须有第一帧或最后一帧
		if mustHaveFirst {
			// 找到前5%时间内得分最高的帧
			earlyThreshold := videoInfo.Duration * 0.05
			var bestEarly *Frame
			for _, f := range group {
				if f.Timestamp <= earlyThreshold {
					if bestEarly == nil || f.Score > bestEarly.Score {
						bestEarly = f
					}
				}
			}
			if bestEarly != nil {
				result = append(result, bestEarly)
				count--
			}
		}

		if mustHaveLast && count > 0 {
			// 找到后5%时间内得分最高的帧
			lateThreshold := videoInfo.Duration * 0.95
			var bestLate *Frame
			for _, f := range group {
				if f.Timestamp >= lateThreshold {
					if bestLate == nil || f.Score > bestLate.Score {
						bestLate = f
					}
				}
			}
			if bestLate != nil {
				result = append(result, bestLate)
				count--
			}
		}

		// 选择剩余的帧
		for _, frame := range group {
			if count <= 0 {
				break
			}

			// 检查是否已选择
			alreadySelected := false
			for _, s := range result {
				if s == frame {
					alreadySelected = true
					break
				}
			}
			if alreadySelected {
				continue
			}

			// 检查与已选择帧的距离和相似度
			tooClose := false
			for _, s := range result {
				timeDiff := math.Abs(frame.Timestamp - s.Timestamp)
				if timeDiff < idealInterval*0.5 {
					// 检查相似度
					hashDiff := hashDifference(frame.Hash, s.Hash)
					if hashDiff < 10 { // 相似度阈值
						tooClose = true
						break
					}
				}
			}

			if !tooClose {
				result = append(result, frame)
				count--
			}
		}

		return result
	}

	// 从各组选择
	selected = append(selected, selectFromGroup(startGroup, startFrames, true, false)...)
	selected = append(selected, selectFromGroup(middleGroup, middleFrames, false, false)...)
	selected = append(selected, selectFromGroup(endGroup, endFrames, false, true)...)

	// 按时间排序
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].Timestamp < selected[j].Timestamp
	})

	return selected
}

// resizeImage 调整图片大小，保持纵横比，最大总像素数为 maxPixels
func resizeImage(img image.Image, maxPixels int) image.Image {
	bounds := img.Bounds()
	origWidth := bounds.Dx()
	origHeight := bounds.Dy()

	// 计算当前像素总数
	currentPixels := origWidth * origHeight

	// 如果当前像素数已经小于或等于最大像素数，不需要调整
	if currentPixels <= maxPixels {
		return img
	}

	// 计算缩放比例
	ratio := math.Sqrt(float64(maxPixels) / float64(currentPixels))

	// 计算新的尺寸
	newWidth := int(float64(origWidth) * ratio)
	newHeight := int(float64(origHeight) * ratio)

	// 确保尺寸至少为1x1
	if newWidth < 1 {
		newWidth = 1
	}
	if newHeight < 1 {
		newHeight = 1
	}

	// 创建新图像
	newImg := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

	// 简单的缩放算法
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			// 映射到原始图像的坐标
			origX := x * origWidth / newWidth
			origY := y * origHeight / newHeight
			newImg.Set(x, y, img.At(origX, origY))
		}
	}

	return newImg
}

// 保存关键帧
func saveKeyFrames(frames []*Frame, outputDir string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}

	for i, frame := range frames {
		filename := filepath.Join(outputDir, fmt.Sprintf("keyframe_%03d_%.3fs.jpg", i+1, frame.Timestamp))

		file, err := os.Create(filename)
		if err != nil {
			return err
		}
		defer file.Close()

		// 调整图片大小，限制总像素数为 100352
		resizedImg := resizeImage(frame.Image, 100352)

		if err := jpeg.Encode(file, resizedImg, &jpeg.Options{Quality: 95}); err != nil {
			return err
		}

		fmt.Printf("保存关键帧: %s (分数: %.2f)\n", filename, frame.Score)
	}

	return nil
}

// ExtractKeyframes 提取视频关键帧的导出函数
func ExtractKeyframes(videoPath, outputDir string) ([]string, error) {
	config := Config{
		VideoPath: videoPath,
		OutputDir: outputDir,
		TempDir:   "temp",
	}

	// 获取视频信息
	fmt.Println("获取视频信息...")
	videoInfo, err := getVideoInfo(config.VideoPath)
	if err != nil {
		return nil, fmt.Errorf("获取视频信息失败: %w", err)
	}

	fmt.Printf("视频时长: %.2f秒\n", videoInfo.Duration)
	fmt.Printf("帧率: %.2f fps\n", videoInfo.FrameRate)
	fmt.Printf("分辨率: %dx%d\n", videoInfo.Width, videoInfo.Height)

	// 获取I帧信息
	fmt.Println("分析I帧...")
	iFrames, err := getIFrames(config.VideoPath)
	if err != nil {
		return nil, fmt.Errorf("获取I帧信息失败: %w", err)
	}

	// 计算需要选择的帧数
	targetFrameCount := calculateFrameCount(videoInfo.Duration)
	fmt.Printf("目标关键帧数量: %d\n", targetFrameCount)

	// 采样帧进行分析
	sampleInterval := videoInfo.Duration / float64(targetFrameCount*5) // 采样5倍于目标数量的帧
	frames := make([]*Frame, 0)

	// 存储所有指标值用于归一化
	var allRGBStdDevs, allEdgeRatios, allLaplacianVars []float64

	fmt.Println("分析视频帧...")
	progressCount := 0

	for t := 0.0; t < videoInfo.Duration; t += sampleInterval {
		progressCount++
		fmt.Printf("\r进度: %d/%d", progressCount, int(videoInfo.Duration/sampleInterval))

		// 提取帧
		img, err := extractFrame(config.VideoPath, t)
		if err != nil {
			continue
		}

		frameIndex := int(t * videoInfo.FrameRate)
		frame := &Frame{
			Index:     frameIndex,
			Timestamp: t,
			IsIFrame:  iFrames[frameIndex],
			Image:     img,
			Hash:      calculatePerceptualHash(img),
		}

		// 计算各项指标
		rgbStdDev := calculateRGBStdDev(img)
		gray := toGrayscale(img)
		edgeRatio := calculateEdgeRatio(gray)
		laplacianVar := calculateLaplacianVariance(gray)

		allRGBStdDevs = append(allRGBStdDevs, rgbStdDev)
		allEdgeRatios = append(allEdgeRatios, edgeRatio)
		allLaplacianVars = append(allLaplacianVars, laplacianVar)

		// 临时存储指标值
		frame.Score = rgbStdDev // 临时使用
		frames = append(frames, frame)
	}

	fmt.Println("\n计算帧分数...")

	// 重新计算所有帧的分数
	for i, frame := range frames {
		scoreFrame(frame, allRGBStdDevs[i], allEdgeRatios[i], allLaplacianVars[i],
			allRGBStdDevs, allEdgeRatios, allLaplacianVars)
	}

	// 选择关键帧
	fmt.Println("选择关键帧...")
	selectedFrames := selectKeyFrames(frames, videoInfo, targetFrameCount)

	s3list, err := SaveTempImageToS3(selectedFrames, outputDir)
	if err != nil {
		return nil, fmt.Errorf("上传关键帧至S3失败: %w", err)
	}

	// 保存关键帧
	fmt.Println("保存关键帧...")
	if err := saveKeyFrames(selectedFrames, config.OutputDir); err != nil {
		return nil, fmt.Errorf("保存关键帧失败: %w", err)
	}

	fmt.Printf("\n完成! 已保存%d个关键帧到 %s\n", len(selectedFrames), config.OutputDir)
	return s3list, nil
}

// 原有的main函数可以保留作为独立运行的入口点
func main() {
	if len(os.Args) < 2 {
		fmt.Println("使用方法: go run keyscore.go <视频文件路径> [输出目录]")
		os.Exit(1)
	}

	videoPath := os.Args[1]
	outputDir := "keyframes"

	if len(os.Args) > 2 {
		outputDir = os.Args[2]
	}

	if _, err := ExtractKeyframes(videoPath, outputDir); err != nil {
		log.Fatal(err)
	}
}
