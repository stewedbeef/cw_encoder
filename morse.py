import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.io import wavfile
import sys
from typing import Optional
import yaml

class MorseCodeConverter():
    # Load once when imported
    with open("morse.yaml") as f:
        morse_code_chart = yaml.safe_load(f)

    def __init__(self, wpm: float = 20, fc: float = 550, fs: float = 44100):
        """
        Initialises an instance of the MorseCodeConverter class

        :param wpm: Words per minute of the Morse code transmission, as measured with the word PARIS in accordance with convention
        :type wpm: float

        :param fc: Carrier frequency of the output wave in Hertz
        :type fc: float

        :param fs: Sampling frequency of the output in Hertz
        :type fs: float
        """
        self.set_params(wpm, fc, fs)
    
    def set_params(
            self,
            wpm: Optional[float] = None,
            fc: Optional[float] = None,
            fs: Optional[float] = None,
    ) -> None:
        """
        Set the wpm and fc params and recalculate other constants

        :param wpm: Words per minute of the Morse code transmission, as measured with the word PARIS in accordance with convention
        :type wpm: Optional[float]

        :param fc: Carrier frequency of the output wave in Hertz
        :type fc: Optional[float]

        :param fs: Sampling frequency of the output in Hertz
        :type fs: Optional[float]
        """
        self.wpm = wpm or self.wpm
        self.fc = fc or self.fc
        self.fs = fs or self.fs

        self.dit = 1.2 / wpm
        self.dah = 3 * self.dit
        self.dit_gap = self.dit # Gap between dits/dahs
        self.letter_gap = self.dah # Gap between letters
        self.word_gap = 2 * self.dah + self.dit # Gap between words
        # When upscaling from the keying waveform in dits to the sampling
        # frequency fs
        self.upscale_factor = round(self.fs * self.dit)

        return self


    def run(self, input: str) -> npt.NDArray[np.float32]:
        """
        Run the Morse code converter

        :param input: Input string
        :type input: str

        :return: The modulated signal (A1W type transmission)
        :rtype: numpy.typing.NDArray[numpy.float32]
        """
        code = self.str_to_code(input)
        key = self.get_key_instructions(code)
        # sig = self.modulate_with_filter(key)
        sig = self.modulate_with_linear_rises(key)
        return sig

    @classmethod
    def get_key_instructions(cls, code: str) -> list[bool]:
        """
        Convert a code transmission into a value corresponding to the audio Morse code signal.
        
        :param code: a string consisting only of '.', '-' and ' '. One space means a letter boundary, and two or more spaces means a word boundary.
        :type code: str

        :return: an array consisting of whether the audio signal should be on or off
        :rtype: list[bool]
        """
        # Notes:
        # A dit is the atomic unit, consisting of one cell
        # A dah consists of three dit lengths
        # The gap between a dit or dah is one dit-length
        # The gap between letters is one dah-length
        # The gap between words is 2*dah-length + dit-length, or 7*dit-length
        key: list[bool] = [False]
        for i, c in enumerate(code):
            if c == ".":
                key.extend([True, False])
            elif c == "-":
                key.extend([True, True, True, False])
            elif c == " ":
                if code[i+1] == " ":
                    # This one must come first so we consume all the spaces
                    continue # next time it will go to the code[i-1] case
                elif code[i-1] == " ":
                    key.extend([False] * 7) # word gap
                else:
                    key.extend([False] * 3) # letter gap
            else:
                raise ValueError(f"`code` has invalid input, saw {c} ({ord(c)}) at index {i}")
        # key is padded with False on the left and right
        return key

    @classmethod
    def str_to_code(cls, input: str) -> str:
        """
        Converts a string into Morse code representation with ".", "-" and " ". Enter prosigns with their letter abbreviations enclosed in angle brackets.

        Unimplemented: If <<> is encountered, put in a left angle bracket. If <> is encountered, put in a right angle bracket.

        :param input: Input string
        :type input: str

        :return: String of ".", "-", " " and "  " representing dits, dahs, and short and long spaces
        :rtype: str
        """
        input = input.strip()
        code: str = ""
        prosign_mode: bool = False
        for i, c in enumerate(input):
            if c == "":
                continue
            elif c.isspace():
                if input[i+1].isspace():
                    continue
                code += "  "
            elif c == "<":
                if prosign_mode:
                    raise ValueError(f"Encountered < while in prosign mode at index {i}")
                prosign_mode = True
                # if prosign_mode:
                #     code += cls.morse_code_chart["<"]
                # elif input[i+1] == ">":
                #     code += cls.morse_code_chart[">"]
                # else:
                #     prosign_mode = True
            elif c == ">":
                if not prosign_mode:
                    raise ValueError(f"Encountered > while not in prosign mode at index {i}")
                prosign_mode = False
            else:
                try:
                    code += cls.morse_code_chart[c.lower()]
                    if not prosign_mode:
                        code += " "
                except KeyError as e:
                    raise ValueError(f"Invalid character in input string, got '{c}' with value {ord(c)} at index {i}") from e
        code = code.rstrip()
        return code
    
    def modulate_with_linear_rises(
            self,
            key: list[bool],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Modules the keyer instructions into an envelope and sound wave output

        :param key: Description
        :type key: list[bool]
        :return: Description
        :rtype: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        """
        # Rise and fall times, in units of number of samples
        t_ramp = round(min(0.01, 0.1*self.dit) * self.fs)

        # Raise cosine ramp
        ramp_up = 0.5 * (1 - np.cos(np.pi * np.arange(t_ramp) / t_ramp))
        ramp_down = ramp_up[::-1]
        # Linear ramp
        # ramp_up = len(t_ramp) / upscale_factor * np.arange(t_ramp)

        envelope = np.repeat(
            np.array(key, dtype=np.float32),
            self.upscale_factor,
        )

        # Don't do the start and end since there won't be a boundary
        for i in range(1, len(key)-1):
            start = i * self.upscale_factor
            end = (i + 1) * self.upscale_factor
            if key[i]:
                if not key[i-1]:
                    envelope[start : start+t_ramp] = ramp_up
                if not key[i+1]:
                    envelope[end-t_ramp : end] = ramp_down
        oscillator = np.sin(
            2 * np.pi * self.fc / self.fs * np.arange(len(envelope)),
            dtype=np.float32,
        )
        output = envelope * oscillator

        # Normalise the output
        output = output / np.max(np.abs(output))
        return envelope, output


    def modulate_with_filter(self, key: list[bool]) -> npt.NDArray[np.float32]:
        """
        Modulates the keyer output into a sound wave

        :param key: Keyer input
        :type key: list[bool]
        :return: Modulated signal (class A1W transmission)
        :rtype: numpy.typing.NDArray[numpy.float32]
        """
        key = np.array(key, dtype=np.float32)

        # Pass through a low pass filter
        # Cutoff freq for filter is 2x to 3x the max freq of morse code, which
        # is 1/dit
        cutoff = 3 / self.dit
        numtaps = 1001 # Should be 1001 to 4001

        # Instantiate a finite impulse response filter
        key = np.repeat(key, self.upscale_factor)
        kernel: npt.NDArray[np.float32] = signal.firwin(
            numtaps, cutoff, window="blackman", fs=self.fs)
        envelope = signal.lfilter(kernel, 1.0, key).astype(np.float32)

        oscillator = np.sin(
            2 * np.pi * self.fc / self.fs * np.arange(len(envelope)),
            dtype=np.float32,
        )
        output = envelope * oscillator

        # Normalise the output
        output = output / np.max(np.abs(output))
        return envelope, output

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Expected input and output files", file=sys.stderr)
        sys.exit()
    with open(sys.argv[1]) as f:
        input = f.read()
    converter = MorseCodeConverter(wpm=25)
    envelope, sound = converter.run(input)
    wavfile.write(sys.argv[2], 44100, np.round(sound * 32767, dtype=np.int16))
