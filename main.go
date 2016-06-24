package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) < 2 {
		dieUsage()
	}

	var err error
	switch os.Args[1] {
	case "train":
		if len(os.Args) != 6 {
			dieUsage()
		}
		rnnFile := os.Args[2]
		compressorFile := os.Args[3]
		wavDir := os.Args[4]
		stepSize, parseErr := strconv.ParseFloat(os.Args[5], 64)
		if parseErr != nil {
			fmt.Fprintln(os.Stderr, "Invalid step size:", os.Args[5])
			os.Exit(2)
		}
		err = Train(rnnFile, compressorFile, wavDir, stepSize)
	case "talk":
		if len(os.Args) != 5 {
			dieUsage()
		}
		rnnFile := os.Args[2]
		outputFile := os.Args[3]
		duration, parseErr := strconv.ParseFloat(os.Args[4], 64)
		if parseErr != nil {
			fmt.Fprintln(os.Stderr, "Invalid duration:", os.Args[4])
			os.Exit(2)
		}
		err = Talk(rnnFile, outputFile, duration)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr,
		"Usage: rnn-talk train <rnn-file> <compressor> <wav dir> <step size>\n"+
			"       rnn-talk talk <rnn-file> <output.wav> <seconds>")
	os.Exit(2)
}
