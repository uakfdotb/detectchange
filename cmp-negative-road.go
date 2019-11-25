package main

import (
	lib "github.com/mitroadmaps/gomapinfer/image"

	"encoding/json"
	"io/ioutil"
	"fmt"
	"os"
	"strings"
)

const PATH = "/data/discover-datasets/2018may10-ken/doha/osmtile-12px/"
const OutPath = "out/"
const OldSatPath = "/data/discover-datasets/2018may10-ken/doha/sat-jpg-old/%s_sat.jpg"
const NewSatPath = "/data/discover-datasets/2018may10-ken/doha/sat-jpg/%s_sat.jpg"
const OldSegPath = "/data/discover-datasets/2018may10-ken/doha/seg-old/%s_seg.png"
const NewSegPath = "/data/discover-datasets/2018may10-ken/doha/seg/%s_seg.png"

func main() {
	files, err := ioutil.ReadDir(PATH)
	if err != nil {
		panic(err)
	}
	ch := make(chan string)
	donech := make(chan bool)
	n := 12
	for i := 0; i < n; i++ {
		go func() {
			for label := range ch {
				process(label)
			}
			donech <- true
		}()
	}
	for _, fi := range files {
		label := strings.Replace(fi.Name(), "_osm.png", "", -1)
		ch <- label
	}
	close(ch)
	for i := 0; i < n; i++ {
		<- donech
	}
}

func process(label string) {
	oldFname := PATH + label + "_osm.png"
	oldSatFname := fmt.Sprintf(OldSatPath, label)
	newSatFname := fmt.Sprintf(NewSatPath, label)
	oldSegFname := fmt.Sprintf(OldSegPath, label)
	newSegFname := fmt.Sprintf(NewSegPath, label)
	if _, err := os.Stat(newSegFname); err != nil {
		fmt.Printf("... skip missing %s\n", label)
		return
	}
	fmt.Println(label)

	// we will take intersection(oldPix, newSeg, NOT oldSeg)
	oldPix := lib.ReadGrayImage(oldFname)
	oldSat := lib.ReadImage(oldSatFname)
	newSat := lib.ReadImage(newSatFname)
	oldSeg := lib.ReadGrayImage(oldSegFname)
	newSeg := lib.ReadGrayImage(newSegFname)

	oldPixBin := lib.Binarize(oldPix, 128)
	oldPixDil := lib.Dilate(oldPixBin, 40)
	oldPixDilForSegMask := lib.Dilate(oldPixBin, 12)

	oldSegBin := lib.Binarize(oldSeg, 16)
	oldSegDil := lib.Dilate(oldSegBin, 10)
	newSegBin := lib.Binarize(newSeg, 128)

	outBin := lib.MakeBinLike(oldSegBin)
	segMaskBin := lib.MakeBinLike(oldSegBin)
	for i := range oldSegBin {
		for j := range oldSegBin[i] {
			segMaskBin[i][j] = newSegBin[i][j] && !oldPixDilForSegMask[i*4][j*4] && !oldSegDil[i][j]
			outBin[i][j] = newSegBin[i][j] && !oldPixDil[i*4][j*4] && !oldSegDil[i][j]
		}
	}

	vec := make([][][3]uint8, len(outBin))
	for i := range outBin {
		vec[i] = make([][3]uint8, len(outBin[i]))
	}
	for i := range vec {
		for j := range vec[i] {
			if oldPixDil[i*4][j*4] {
				vec[i][j][0] = 255
			}
			if oldSegDil[i][j] {
				vec[i][j][1] = 255
			}
			if newSegBin[i][j] {
				vec[i][j][2] = 255
			}
		}
	}
	outBinImg := lib.Unbinarize(outBin, 255)
	segMaskImg := lib.Unbinarize(segMaskBin, 255)

	seenMask := lib.MakeBinLike(outBin)
	for i := range seenMask {
		for j := range seenMask[i] {
			if seenMask[i][j] || !outBin[i][j] {
				continue
			}
			members := lib.Floodfill(outBin, seenMask, i, j)
			sx, sy, ex, ey := lib.PointsToRect(members)
			threshold := 24
			if ex - sx < threshold && ey - sy < threshold {
				continue
			}
			fmt.Printf("... got at %s (%d, %d) to (%d, %d)\n", label, sx, sy, ex, ey)
			cx := (sx+ex)/2
			cy := (sy+ey)/2

			var centerMember [2]int
			var bestDistance int = -1
			var confidenceSum int
			for _, member := range members {
				dx := member[0] - cx
				dy := member[1] - cy
				d := dx * dx + dy * dy
				if bestDistance == -1 || d < bestDistance {
					centerMember = member
				}
				confidenceSum += int(newSeg[member[0]][member[1]])
			}
			confidenceAvg := confidenceSum / len(members)

			show(label, oldSat, newSat, oldPix, oldSeg, newSeg, outBinImg, segMaskImg, vec, cx, cy, centerMember, confidenceAvg)
		}
	}
}

func show(label string, oldSat [][][3]uint8, newSat [][][3]uint8, oldPix [][]uint8, oldSeg [][]uint8, newSeg [][]uint8, outBinImg [][]uint8, segMaskImg [][]uint8, vec [][][3]uint8, cx int, cy int, centerMember [2]int, confidenceAvg int) {
	cx *= 4
	cy *= 4
	if cx < 192 {
		cx = 192
	} else if cx > 3904 {
		cx = 3904
	}
	if cy < 192 {
		cy = 192
	} else if cy > 3904 {
		cy = 3904
	}
	sx := cx-192
	sy := cy-192
	ex := cx+192
	ey := cy+192
	oldCrop := lib.Crop(oldSat, sx, sy, ex, ey)
	newCrop := lib.Crop(newSat, sx, sy, ex, ey)
	oldSegCrop := lib.CropGray(oldSeg, sx/4, sy/4, ex/4, ey/4)
	newSegCrop := lib.CropGray(newSeg, sx/4, sy/4, ex/4, ey/4)
	osmCrop := lib.CropGray(oldPix, sx, sy, ex, ey)
	vecCrop := lib.Crop(vec, sx/4, sy/4, ex/4, ey/4)
	segMaskCrop := lib.CropGray(segMaskImg, sx/4, sy/4, ex/4, ey/4)

	prefix := fmt.Sprintf(OutPath + "/%s.%d-%d.%d", label, sx, sy, confidenceAvg)
	lib.WriteImage(prefix + ".old.png", oldCrop)
	lib.WriteImage(prefix + ".new.png", newCrop)
	lib.WriteGrayImage(prefix + ".oldseg.png", oldSegCrop)
	lib.WriteGrayImage(prefix + ".newseg.png", newSegCrop)
	lib.WriteGrayImage(prefix + ".zosm.png", osmCrop)
	lib.WriteImage(prefix + ".vec.png", vecCrop)
	lib.WriteGrayImage(prefix + ".segmask.png", segMaskCrop)

	bytes, err := json.Marshal(centerMember)
	if err != nil {
		panic(err)
	}
	if err := ioutil.WriteFile(prefix + ".json", bytes, 0644); err != nil {
		panic(err)
	}
}
