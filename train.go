package main

import (
	"fmt"
	"io/ioutil"
	"log"

	"github.com/unixpickle/eigensongs"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	outputPermissions           = 0755
	validationBatchSize         = 10
	trainingBatchSize           = 10
	trainingEquilibrationMemory = 0.9
	trainingHeadSize            = 50
	trainingTailSize            = 20
	trainingMaxLanes            = 25
	trainingDamping             = 1e-4
)

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

	log.Println("Training LSTM on", samples.Samples.Len(), "samples...")
	trainWithSamples(talker, samples, stepSize)

	data, err := talker.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(rnnFile, data, outputPermissions)
}

func trainWithSamples(talker *Talker, s *SampleInfo, step float64) {
	talker.SetDropout(false)
	//talker.SetDropout(true)
	//defer talker.SetDropout(false)

	costFunc := neuralnet.MeanSquaredCost{}
	gradienter := &neuralnet.AdaGrad{
		Gradienter: &rnn.BPTT{
			Learner:  talker.Block,
			CostFunc: costFunc,
			MaxLanes: trainingMaxLanes,
			//HeadSize: trainingHeadSize,
			//TailSize: trainingTailSize,
		},
		Damping: trainingDamping,
	}

	var epoch int
	neuralnet.SGDInteractive(gradienter, s.Samples, step, trainingBatchSize, func() bool {
		//talker.SetDropout(false)
		//defer talker.SetDropout(true)

		runner := &rnn.Runner{Block: talker.Block}
		cost := runner.TotalCost(validationBatchSize, s.Samples, costFunc)
		log.Printf("Epoch %d: cost=%f", epoch, cost)

		epoch++
		return true
	})
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
