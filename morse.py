import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.io import wavfile
import sys
import yaml

class MorseCodeConverter():
    # Load once when imported
    with open("morse.yaml") as f:
        morse_code_chart = yaml.safe_load(f)

    def __init__(self, wpm: float=20, fc: float=550, fs: float=44100):
        """
        Initialises an instance of the MorseCodeConverter class

        :param wpm: Words per minute of the Morse code transmission, as measured with the word PARIS in accordance with convention
        :type wpm: float

        :param fc: Carrier frequency of the output wave in Hertz
        :type fc: float
        """
        self.set_params(wpm, fc, fs)
    
    def set_params(self, wpm: float, fc: float, fs: float) -> None:
        """
        Set the wpm and fc params and recalculate other constants

        :param wpm: Words per minute of the Morse code transmission, as measured with the word PARIS in accordance with convention
        :type wpm: float

        :param fc: Carrier frequency of the output wave in Hertz
        :type fc: float

        :param fs: Sampling frequency of output
        :type fs: float
        """
        self.wpm = wpm
        self.fc = fc
        self.fs = fs

        self.dit = 1.2 / wpm
        self.dah = 3 * self.dit
        self.dit_gap = self.dit # Gap between dits/dahs
        self.letter_gap = self.dah # Gap between letters
        self.word_gap = 2 * self.dah + self.dit # Gap between words

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
        sig = self.modulate(key)
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
        key: list[bool] = []
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
        return key[:-1] if not key[-1] else key

    @classmethod
    def str_to_code(cls, input: str) -> str:
        """
        Converts a string into Morse code representation with ".", "-" and " "

        :param input: Input string
        :type input: str

        :return: String of ".", "-", " " and "  " representing dits, dahs, and short and long spaces
        :rtype: str
        """
        input = input.strip()
        code: str = ""
        for i, c in enumerate(input):
            if c == "":
                continue
            elif c == " ":
                if input[i+1] == " ":
                    continue
                code += "  "
            elif c == "\n":
                if input[i+1] == "\n":
                    continue
                elif input[i-1] == "\n":
                    # New section
                    code += "  " + cls.morse_code_chart["="] + "  "
                else:
                    # Newline
                    code += "  " + cls.morse_code_chart["/"] + "  "
            else:
                try:
                    code += cls.morse_code_chart[c.lower()] + " "
                except KeyError as e:
                    raise ValueError(f"Invalid character in input string, got '{c}' with value {ord(c)} at index {i}") from e
        # Append AR to signal end
        code = code.rstrip() + cls.morse_code_chart["+"]
        return code

    def modulate(self, key: list[bool]) -> npt.NDArray[np.float32]:
        """
        Modulates the keyer output into a sound wave

        :param key: Keyer input
        :type key: list[bool]
        :return: Modulated signal (class A1W transmission)
        :rtype: numpy.typing.NDArray[numpy.float32]
        """
        key = np.array(key).astype(np.float32)
        # Pass through a low pass filter
        # Cutoff freq for filter is 2x to 3x the max freq of morse code, which
        # is 1/dit
        cutoff = 3 / self.dit
        numtaps = 1001 # Should be 1001 to 4001

        # Instantiate a finite impulse response filter
        # print("Starting filtering", flush=True)
        kernel: npt.NDArray[np.float32] = signal.firwin(numtaps, cutoff, window="blackman", fs=self.fs)
        key = np.repeat(key, round(self.fs * self.dit))
        intermediate = signal.lfilter(kernel, 1.0, key).astype(np.float32)
        # print("Done filtering", flush=True)
        oscillator = np.sin(2 * np.pi * self.fc / self.fs * np.arange(len(intermediate))).astype(np.float32)
        output = intermediate * oscillator
        # Normalise the output
        output = output / np.max(np.abs(output))
        return output

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Expected input and output files", file=sys.stderr)
        sys.exit()
    with open(sys.argv[1]) as f:
        input = f.read()
    converter = MorseCodeConverter()
    wavfile.write(sys.argv[2], 44100, (converter.run(input) * 32767).astype(np.int16))
