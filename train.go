package main

import (
	"fmt"
	"io/ioutil"

	"github.com/unixpickle/eigensongs"
	"github.com/unixpickle/serializer"
)

const outputPermissions = 0755

func Train(rnnFile, compressorFile, wavDir string, stepSize float64) error {
	compressor, err := readCompressor(compressorFile)
	if err != nil {
		return err
	}
	samples, err := ReadSamples(wavDir, compressor)
	if err != nil {
		return err
	}

	var talker *Talker
	talkerData, err := ioutil.ReadFile(rnnFile)
	if err == nil {
		talker, err = DeserializeTalker(talkerData)
		if err != nil {
			return err
		}
	} else {
		talker = NewTalker(samples, compressor)
	}

	// TODO: train here.

	data, err := talker.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(rnnFile, data, outputPermissions)
}

func readCompressor(compressorFile string) (*eigensongs.Compressor, error) {
	compressorData, err := ioutil.ReadFile(compressorFile)
	if err != nil {
		return nil, err
	}
	fileObj, err := serializer.DeserializeWithType(compressorData)
	if err != nil {
		return nil, err
	}
	compressor, ok := fileObj.(*eigensongs.Compressor)
	if !ok {
		return nil, fmt.Errorf("invalid compressor type: %T", fileObj)
	}
	return compressor, nil
}
