package main

import (
	"io/ioutil"
	"log"

	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	outputPermissions = 0755
	trainingBatchSize = 1
)

func Train(rnnFile, wavDir string, stepSize float64) error {
	log.Println("Loading samples...")
	samples, err := ReadSampleSet(wavDir)
	if err != nil {
		return err
	}

	log.Println("Creating/loading talker...")
	var talker *Talker
	talkerData, err := ioutil.ReadFile(rnnFile)
	if err == nil {
		talker, err = DeserializeTalker(talkerData)
		if err != nil {
			return err
		}
	} else {
		talker = NewTalker()
	}

	log.Println("Training RNN on", samples.Len(), "samples...")
	trainWithSamples(talker, samples, stepSize)

	data, err := talker.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(rnnFile, data, outputPermissions)
}

func trainWithSamples(talker *Talker, s SampleSet, step float64) {
	talker.SetDropout(true)
	defer talker.SetDropout(false)

	costFunc := neuralnet.DotCost{}
	gradienter := &sgd.Adam{
		Gradienter: &seqtoseq.Gradienter{
			SeqFunc:  &rnn.BlockSeqFunc{talker.Block},
			Learner:  talker.Block,
			CostFunc: costFunc,
		},
	}

	var iter int
	var last sgd.SampleSet
	sgd.SGDMini(gradienter, s, step, trainingBatchSize, func(m sgd.SampleSet) bool {
		talker.SetDropout(false)
		defer talker.SetDropout(true)
		var lastCost float64
		if last != nil {
			lastCost = seqtoseq.TotalCostBlock(talker.Block, 1, last, costFunc)
		}
		last = m.Copy()
		cost := seqtoseq.TotalCostBlock(talker.Block, 1, m, costFunc)
		log.Printf("Iteration %d: cost=%f last=%f", iter, cost, lastCost)
		iter++
		return true
	})
}
