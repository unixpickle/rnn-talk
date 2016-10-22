package main

import (
	"sync"

	"github.com/unixpickle/wav"
)

type sampleCache struct {
	lock  sync.Mutex
	cache map[string]wav.Sound
}

func newSampleCache() *sampleCache {
	return &sampleCache{cache: map[string]wav.Sound{}}
}

func (s *sampleCache) readFile(path string) wav.Sound {
	s.lock.Lock()
	defer s.lock.Unlock()

	item, ok := s.cache[path]
	if ok {
		cp := wav.NewPCM8Sound(item.Channels(), item.SampleRate())
		wav.Append(cp, item)
		return cp
	}

	// TODO: evict sound files if we get too full.

	contents, err := wav.ReadSoundFile(path)
	if err != nil {
		panic(err)
	}
	s.cache[path] = contents
	cp := wav.NewPCM8Sound(contents.Channels(), contents.SampleRate())
	wav.Append(cp, contents)
	return cp
}
