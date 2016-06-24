package main

import (
	"errors"

	"github.com/unixpickle/eigensongs"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const talkerSerializerType = "github.com/unixpickle/rnn-talk.Talker"

var (
	hiddenLayerSizes    = []int{300}
	hiddenLayerDropouts = []float64{0.5}

	invalidSliceErr = errors.New("invalid deserialized slice")
)

type Talker struct {
	Block      rnn.StackedBlock
	Compressor *eigensongs.Compressor
}

func NewTalker(samples neuralnet.SampleSet, comp *eigensongs.Compressor) *Talker {
	_, compressedSize := comp.Dims()

	var stackedBlock rnn.StackedBlock

	mean, stddev := SampleStats(samples)
	normalizeNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -mean, Scale: 1 / stddev},
	}
	stackedBlock = append(stackedBlock, rnn.NewNetworkBlock(normalizeNet, 0))

	for i, layerSize := range hiddenLayerSizes {
		var inputSize int
		if i > 0 {
			inputSize = hiddenLayerSizes[i-1]
		} else {
			inputSize = compressedSize
		}
		layer := rnn.NewLSTM(inputSize, layerSize)
		initializeBiases(layer)
		stackedBlock = append(stackedBlock, layer)

		dropoutNetwork := neuralnet.Network{
			&neuralnet.DropoutLayer{
				KeepProbability: hiddenLayerDropouts[i],
				Training:        false,
			},
		}
		dropoutBlock := rnn.NewNetworkBlock(dropoutNetwork, 0)
		stackedBlock = append(stackedBlock, dropoutBlock)
	}
	outputNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  hiddenLayerSizes[len(hiddenLayerSizes)-1],
			OutputCount: compressedSize,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outputNet.Randomize()
	outputBlock := rnn.NewNetworkBlock(outputNet, 0)
	stackedBlock = append(stackedBlock, outputBlock)

	return &Talker{
		Block:      stackedBlock,
		Compressor: comp,
	}
}

func DeserializeTalker(d []byte) (*Talker, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}

	if len(slice) != 2 {
		return nil, invalidSliceErr
	}

	block, ok := slice[0].(rnn.StackedBlock)
	if !ok {
		return nil, invalidSliceErr
	}
	comp, ok := slice[1].(*eigensongs.Compressor)
	if !ok {
		return nil, invalidSliceErr
	}

	return &Talker{
		Block:      block,
		Compressor: comp,
	}, nil
}

func (t *Talker) Serialize() ([]byte, error) {
	slice := []serializer.Serializer{t.Block, t.Compressor}
	return serializer.SerializeSlice(slice)
}

func (t *Talker) SerializerType() string {
	return talkerSerializerType
}

func initializeBiases(layer *rnn.LSTM) {
	inputBiases := layer.Parameters()[3]
	for i := range inputBiases.Vector {
		inputBiases.Vector[i] = -1
	}
	outputBiases := layer.Parameters()[7]
	for i := range outputBiases.Vector {
		outputBiases.Vector[i] = -2
	}
}

func init() {
	serializer.RegisterDeserializer(talkerSerializerType,
		func(d []byte) (serializer.Serializer, error) {
			return DeserializeTalker(d)
		})
}
