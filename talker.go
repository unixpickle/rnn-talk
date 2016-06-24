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
	SampleRate int
	Channels   int

	Min, Max float64
}

func NewTalker(info *SampleInfo, comp *eigensongs.Compressor) *Talker {
	_, compressedSize := comp.Dims()

	var stackedBlock rnn.StackedBlock

	mean, stddev := SampleStats(info.Samples)
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
		&neuralnet.Sigmoid{},
	}
	outputNet.Randomize()
	outputBlock := rnn.NewNetworkBlock(outputNet, 0)
	stackedBlock = append(stackedBlock, outputBlock)

	return &Talker{
		Block:      stackedBlock,
		Compressor: comp,
		SampleRate: info.SampleRate,
		Channels:   info.Channels,

		Min: info.Min,
		Max: info.Max,
	}
}

func DeserializeTalker(d []byte) (*Talker, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}

	if len(slice) != 6 {
		return nil, invalidSliceErr
	}

	block, ok1 := slice[0].(rnn.StackedBlock)
	comp, ok2 := slice[1].(*eigensongs.Compressor)
	sampleRate, ok3 := slice[2].(serializer.Int)
	channels, ok4 := slice[3].(serializer.Int)
	min, ok5 := slice[4].(serializer.Float64)
	max, ok6 := slice[5].(serializer.Float64)
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 || !ok6 {
		return nil, invalidSliceErr
	}

	return &Talker{
		Block:      block,
		Compressor: comp,
		SampleRate: int(sampleRate),
		Channels:   int(channels),
		Min:        float64(min),
		Max:        float64(max),
	}, nil
}

// SetDropout enables or disables random dropout.
func (t *Talker) SetDropout(useDropout bool) {
	for i := 2; i < len(t.Block); i += 2 {
		networkBlock := t.Block[i].(*rnn.NetworkBlock)
		network := networkBlock.Network()
		dropout := network[0].(*neuralnet.DropoutLayer)
		dropout.Training = useDropout
	}
}

// SetTraining enables or disables training mode.
// In training mode, the output of the network is
// not run through a sigmoid activation function.
func (t *Talker) SetTraining(training bool) {
	outNet := t.Block[len(t.Block)-1].(*rnn.NetworkBlock).Network()
	if training {
		outNet = outNet[:1]
	} else {
		outNet = append(outNet[:1], &neuralnet.Sigmoid{})
	}
	t.Block[len(t.Block)-1] = rnn.NewNetworkBlock(outNet, 0)
}

func (t *Talker) Serialize() ([]byte, error) {
	slice := []serializer.Serializer{t.Block, t.Compressor,
		serializer.Int(t.SampleRate), serializer.Int(t.Channels),
		serializer.Float64(t.Min), serializer.Float64(t.Max)}
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
