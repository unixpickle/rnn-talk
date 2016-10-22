package main

import (
	"errors"
	"io/ioutil"
	"math"
	"math/rand"

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
	talker.SetDropout(false)

	// TODO: support priming here.
	if primingFile != "" {
		return errors.New("priming not yet implemented")
	}

	runner := &rnn.Runner{Block: talker.Block}

	var output []wav.Sample
	lastOutput := discreteSample(0)
	for i := 0; i < int(seconds*float64(sampleRate)); i++ {
		nextOut := runner.StepTime(lastOutput)
		outSample := chooseIndex(nextOut)
		lastOutput = make(linalg.Vector, len(lastOutput))
		lastOutput[outSample] = 1
		output = append(output, continuousSample(outSample))
	}

	outSound := wav.NewPCM8Sound(1, sampleRate)
	outSound.SetSamples(output)
	return wav.WriteFile(outSound, outputFile)
}

func chooseIndex(v linalg.Vector) int {
	num := rand.Float64()
	for i, x := range v {
		num -= math.Exp(x)
		if num <= 0 {
			return i
		}
	}
	return len(v) - 1
}

/*
func primeTalker(t *Talker, r *rnn.Runner, primingFile string) error {
	sound, err := wav.ReadSoundFile(primingFile)
	if err != nil {
		return errors.New("failed to read priming file: " + err.Error())
	}
	sample := soundToSample(sound, t.Compressor)
	for _, input := range sample.Inputs {
		for i, x := range input {
			input[i] = (x - t.Min) / (t.Max - t.Min)
		}
		r.StepTime(input)
	}
	return nil
}
*/
