package main

import (
	"errors"
	"io/ioutil"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/wav"
	"github.com/unixpickle/weakai/rnn"
)

func Echo(rnnFile, inputFile, outputFile string) error {
	talkerData, err := ioutil.ReadFile(rnnFile)
	if err != nil {
		return err
	}
	talker, err := DeserializeTalker(talkerData)
	if err != nil {
		return err
	}
	talker.SetTraining(false)
	talker.SetDropout(false)

	runner := &rnn.Runner{Block: talker.Block}

	inputSound, err := wav.ReadSoundFile(inputFile)
	if err != nil {
		return errors.New("failed to read input file: " + err.Error())
	}
	inputSample := soundToSample(inputSound, talker.Compressor)

	chunkSize, _ := talker.Compressor.Dims()

	var outputSamples []wav.Sample
	tempSamples := make([]wav.Sample, chunkSize)

	for _, input := range inputSample.Inputs {
		for i, x := range input {
			input[i] = (x - talker.Min) / (talker.Max - talker.Min)
		}
		output := runner.StepTime(input)
		for i, x := range output {
			output[i] = talker.Min + x*(talker.Max-talker.Min)
		}
		mat := &linalg.Matrix{
			Rows: 1,
			Cols: len(output),
			Data: output,
		}
		decompressed := talker.Compressor.Decompress(mat)
		for i, x := range decompressed.Data {
			tempSamples[i] = wav.Sample(x)
		}
		outputSamples = append(outputSamples, tempSamples...)
	}

	outSound := wav.NewPCM8Sound(talker.Channels, talker.SampleRate)
	outSound.SetSamples(outputSamples)
	return wav.WriteFile(outSound, outputFile)
}
