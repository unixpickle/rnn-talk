package main

import (
	"errors"
	"io/ioutil"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/wav"
	"github.com/unixpickle/weakai/rnn"
)

func Talk(rnnFile, outputFile string, seconds float64, primingFile string) error {
	talkerData, err := ioutil.ReadFile(rnnFile)
	if err != nil {
		return err
	}
	talker, err := DeserializeTalker(talkerData)
	if err != nil {
		return err
	}
	talker.SetTraining(false)

	runner := &rnn.Runner{Block: talker.Block}
	if primingFile != "" {
		talker.SetDropout(false)
		if err := primeTalker(talker, runner, primingFile); err != nil {
			return err
		}
	} else {
		talker.SetDropout(true)
	}

	chunkSize, _ := talker.Compressor.Dims()
	count := int(float64(talker.Channels*talker.SampleRate) * seconds / float64(chunkSize))

	var output []wav.Sample
	lastOutput := make(linalg.Vector, talker.Block.StateSize())
	tempSamples := make([]wav.Sample, chunkSize)
	for i := 0; i < count; i++ {
		lastOutput = runner.StepTime(lastOutput)
		for i, x := range lastOutput {
			lastOutput[i] = talker.Min + x*(talker.Max-talker.Min)
		}
		mat := &linalg.Matrix{
			Rows: 1,
			Cols: len(lastOutput),
			Data: lastOutput,
		}
		decompressed := talker.Compressor.Decompress(mat)
		for i, x := range decompressed.Data {
			tempSamples[i] = wav.Sample(x)
		}
		output = append(output, tempSamples...)
	}

	outSound := wav.NewPCM8Sound(talker.Channels, talker.SampleRate)
	outSound.SetSamples(output)
	return wav.WriteFile(outSound, outputFile)
}

func primeTalker(t *Talker, r *rnn.Runner, primingFile string) error {
	sound, err := wav.ReadSoundFile(primingFile)
	if err != nil {
		return errors.New("failed to read priming file: " + err.Error())
	}
	sample := soundToSample(sound, t.Compressor)
	for _, input := range sample.Inputs {
		for i, x := range input {
			input[i] = t.Min + x*(t.Max-t.Min)
		}
		r.StepTime(input)
	}
	return nil
}
