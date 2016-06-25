package main

import (
	"io/ioutil"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/wav"
	"github.com/unixpickle/weakai/rnn"
)

func Talk(rnnFile, outputFile string, seconds float64) error {
	talkerData, err := ioutil.ReadFile(rnnFile)
	if err != nil {
		return err
	}
	talker, err := DeserializeTalker(talkerData)
	if err != nil {
		return err
	}
	talker.SetDropout(true)

	chunkSize, _ := talker.Compressor.Dims()
	count := int(float64(talker.Channels*talker.SampleRate) * seconds / float64(chunkSize))

	var output []wav.Sample

	runner := &rnn.Runner{Block: talker.Block}
	lastOutput := make(linalg.Vector, talker.Block.StateSize())
	tempSamples := make([]wav.Sample, chunkSize)
	for i := 0; i < count; i++ {
		lastOutput = runner.StepTime(lastOutput)
		mat := &linalg.Matrix{
			Rows: 1,
			Cols: len(lastOutput),
			Data: lastOutput,
		}
		decompressed := talker.Compressor.Decompress(mat)
		for i, x := range decompressed.Data {
			tempSamples[i] = wav.Sample(talker.Min + x*(talker.Max-talker.Min))
		}
		output = append(output, tempSamples...)
	}

	outSound := wav.NewPCM8Sound(talker.Channels, talker.SampleRate)
	outSound.SetSamples(output)
	return wav.WriteFile(outSound, outputFile)
}
