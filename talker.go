package main

import (
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

var (
	hiddenLayerSizes    = []int{100}
	hiddenLayerDropouts = []float64{1}
)

func init() {
	var t Talker
	serializer.RegisterTypedDeserializer(t.SerializerType(), DeserializeTalker)
}

type Talker struct {
	Block rnn.StackedBlock
}

func NewTalker() *Talker {
	var stackedBlock rnn.StackedBlock
	for i, layerSize := range hiddenLayerSizes {
		var inputSize int
		if i > 0 {
			inputSize = hiddenLayerSizes[i-1]
		} else {
			inputSize = len(discreteSample(0))
		}
		layer := rnn.NewLSTM(inputSize, layerSize)
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
			OutputCount: len(discreteSample(0)),
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outputNet.Randomize()
	outputBlock := rnn.NewNetworkBlock(outputNet, 0)
	stackedBlock = append(stackedBlock, outputBlock)

	return &Talker{
		Block: stackedBlock,
	}
}

func DeserializeTalker(d []byte) (*Talker, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &Talker{Block: block}, nil
}

// SetDropout enables or disables random dropout.
func (t *Talker) SetDropout(useDropout bool) {
	for i := 1; i < len(t.Block); i += 2 {
		networkBlock := t.Block[i].(*rnn.NetworkBlock)
		network := networkBlock.Network()
		dropout := network[0].(*neuralnet.DropoutLayer)
		dropout.Training = useDropout
	}
}

func (t *Talker) Serialize() ([]byte, error) {
	return t.Block.Serialize()
}

func (t *Talker) SerializerType() string {
	return "github.com/unixpickle/rnn-talk.Talker"
}
