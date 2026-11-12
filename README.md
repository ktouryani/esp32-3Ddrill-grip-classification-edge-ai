# esp32-3Ddrill-grip-classification-edge-ai
Real-time grip classification on an ESP32 with an INT8 neural network. Master's thesis project at TUHH with a 3D-printed drill prototype, C++ firmware and a Python training pipeline.

<img width="366" height="530" alt="tool_internal_setup" src="https://github.com/user-attachments/assets/8399dbe8-749a-4dde-b935-be2119eac6df" />
<img width="427" height="567" alt="3dtool" src="https://github.com/user-attachments/assets/ca444cce-3c84-4d8d-8f8e-864db05e96b4" />


## Results
- 89.3% offline accuracy on 5 grip classes
- 92% agreement between the ESP32 firmware and the Python reference pipeline, after closing a 15%+ gap
- Data from 21 participants, 4,279 samples

## How it works
1. 1. The 3D-printed drill housing contains an ESP32, an IMU and an ERM (eccentric rotating mass) vibration motor. The motor excites the housing with vibrations, and the way a hand grips the handle changes those vibrations, which the IMU measures.
2. An IMU on the drill prototype is read by the ESP32 at 2 to 3 kHz, using interrupt-driven scheduling and a ring buffer.
3. The firmware computes FFT-based features on the device.
4. A hybrid CNN/MLP, trained in Python with TensorFlow and quantized to INT8, classifies the grip on the chip.
5. The ESP32 and a Python script exchange data over a serial link using pyserial.

## Debugging story
The deployed firmware first disagreed with the Python results by more than 15%. I compared both codebases line by line and found three bugs, in the FFT windowing, in power versus magnitude, and in the band-power formula. After fixing them, agreement rose to 92%.

## Repository contents
- `firmware/` ESP32 C++ code (scheduler, ring buffer, feature extraction)
- `python/` training, quantization and evaluation scripts
- `docs/` Thesis Report


## Hardware
ESP32 [WROOM-32], IMU [LSM6DSOX], ERM vibrational motor that excites the housing, all inside a 3D-printed drill prototype.

Note: the raw participant data is not included.
