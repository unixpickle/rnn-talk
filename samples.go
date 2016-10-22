package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/wav"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	sampleDuration = time.Second * 3
	sampleRate     = 8000
)

var errNoAudioFiles = errors.New("no audio files")
var soundCache = newSampleCache()

// SampleInfo stores informatino about a sample.
type SampleInfo struct {
	Path  string
	Start time.Duration
	End   time.Duration
}

// A SampleSet lazily loads audio snippets from a sample
// directory full of WAV files.
type SampleSet []SampleInfo

// ReadSampleSet reads information about all of the WAV
// files in a directory and creates a SampleSet out of
// said information.
func ReadSampleSet(wavDir string) (SampleSet, error) {
	headers, err := readHeaders(wavDir)
	if err != nil {
		return nil, err
	} else if len(headers) == 0 {
		return nil, errNoAudioFiles
	}

	var res SampleSet
	for path, header := range headers {
		dur := header.Duration()
		for t := time.Duration(0); t+sampleDuration <= dur; t += sampleDuration {
			res = append(res, SampleInfo{
				Path:  path,
				Start: t,
				End:   t + sampleDuration,
			})
		}
	}

	return res, nil
}

// Len returns the number of samples.
func (s SampleSet) Len() int {
	return len(s)
}

// Swap swaps the samples at two indices.
func (s SampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// GetSample reads a sample as PCM data and returns the
// data as a seqtoseq.Sample.
func (s SampleSet) GetSample(i int) interface{} {
	sample := s[i]
	sound := soundCache.readFile(sample.Path)
	wav.Crop(sound, sample.Start, sample.End)
	rateConv := wav.NewPCM8Sound(1, sampleRate)
	wav.Append(rateConv, sound)

	var res seqtoseq.Sample
	for _, pcmSample := range rateConv.Samples()[:len(rateConv.Samples())-1] {
		res.Inputs = append(res.Inputs, discreteSample(pcmSample))
	}
	for _, pcmSample := range rateConv.Samples()[1:] {
		res.Outputs = append(res.Outputs, discreteSample(pcmSample))
	}
	return res
}

// Subset returns a subset of the sample set.
func (s SampleSet) Subset(i, j int) sgd.SampleSet {
	return s[i:j]
}

// Copy returns a copy of the sample set.
func (s SampleSet) Copy() sgd.SampleSet {
	res := make(SampleSet, len(s))
	copy(res, s)
	return res
}

func readHeaders(dir string) (map[string]*wav.Header, error) {
	contents, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	res := map[string]*wav.Header{}
	for _, obj := range contents {
		if strings.HasPrefix(obj.Name(), ".") ||
			!strings.HasSuffix(obj.Name(), ".wav") {
			continue
		}
		p := filepath.Join(dir, obj.Name())
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		header, err := wav.ReadHeader(f)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("read %s: %s", p, err)
		}
		res[p] = header
	}

	return res, nil
}

func discreteSample(s wav.Sample) linalg.Vector {
	outVec := make(linalg.Vector, 128)
	outVec[int(s*0x40+0x40)] = 1
	return outVec
}

func continuousSample(i int) wav.Sample {
	return (wav.Sample(i) / 0x40) - 0x40
}
