// AudioWorklet: forwards each 128-sample block of channel 0 with its context sample frame index.
class PcmRecorder extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length) {
      // currentFrame is the context frame of the first sample in this render quantum
      this.port.postMessage({ frame: currentFrame, samples: new Float32Array(input[0]) });
    }
    return true;
  }
}
registerProcessor("pcm-recorder", PcmRecorder);
