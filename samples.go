package main

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"
	"time"

	"github.com/unixpickle/eigensongs"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/wav"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const sampleDuration = time.Minute

// ReadSamples reads all the WAV files from a sample
// directory, compresses them with the compressor,
// splits them up into reasonably sized chunks, and
// returns the resulting samples.
func ReadSamples(wavDir string, comp *eigensongs.Compressor) (neuralnet.SampleSet, error) {
	sounds, err := readSounds(wavDir)
	if err != nil {
		return nil, err
	}
	var res neuralnet.SliceSampleSet
	for _, sound := range choppedSounds(sounds) {
		res = append(res, soundToSample(sound, comp))
	}
	return res, nil
}

// SampleStats returns the standard deviation and mean
// value of sample inputs.
func SampleStats(samples neuralnet.SampleSet) (mean, stddev float64) {
	var count float64
	for i := 0; i < samples.Len(); i++ {
		sample := samples.GetSample(i).(rnn.Sequence)
		for _, input := range sample.Inputs {
			for _, x := range input {
				mean += x
				count += 1
			}
		}
	}
	if count == 0 {
		return
	}
	mean /= count
	for i := 0; i < samples.Len(); i++ {
		sample := samples.GetSample(i).(rnn.Sequence)
		for _, input := range sample.Inputs {
			for _, x := range input {
				stddev += (x - mean) * (x - mean)
			}
		}
	}
	stddev /= count
	return
}

func readSounds(dir string) ([]wav.Sound, error) {
	contents, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var sounds []wav.Sound
	for _, obj := range contents {
		if strings.HasPrefix(obj.Name(), ".") {
			continue
		}
		p := filepath.Join(dir, obj.Name())
		sound, err := wav.ReadSoundFile(p)
		if err != nil {
			return nil, fmt.Errorf("error reading %s: %s", p, err)
		}
		sounds = append(sounds, sound)
	}

	return sounds, nil
}

func choppedSounds(sounds []wav.Sound) []wav.Sound {
	var res []wav.Sound
	for _, fullSound := range sounds {
		duration := fullSound.Duration()
		for t := time.Duration(0); t < duration; t += sampleDuration {
			cropped := fullSound.Clone()
			wav.Crop(cropped, t, t+sampleDuration)
			res = append(res, cropped)
		}
	}
	return res
}

func soundToSample(sound wav.Sound, comp *eigensongs.Compressor) rnn.Sequence {
	var res rnn.Sequence

	chunkSize, compressedSize := comp.Dims()
	samples := sound.Samples()

	floatChunk := make(linalg.Vector, chunkSize)
	floatMat := &linalg.Matrix{
		Rows: 1,
		Cols: len(floatChunk),
		Data: floatChunk,
	}
	for i := 0; i < len(samples)-chunkSize; i += chunkSize {
		chunk := sound.Samples()[i : i+chunkSize]
		for j, x := range chunk {
			floatChunk[j] = float64(x)
		}
		compressed := comp.Compress(floatMat)
		if i > 0 {
			res.Inputs = append(res.Inputs, res.Outputs[len(res.Outputs)-1])
		} else {
			blank := make(linalg.Vector, compressedSize)
			res.Inputs = append(res.Inputs, blank)
		}
		res.Outputs = append(res.Outputs, compressed.Data)
	}

	return res
}
