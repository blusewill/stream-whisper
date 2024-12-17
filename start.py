import speech_recognition as sr
from faster_whisper import WhisperModel
import inquirer
import os.path
import time
import json
import os

PWD = os.getcwd()
if os.name == "nt":
    with open(f"{PWD}\\config.json", "r") as config_file:
        config = json.load(config_file)
else:
    with open(f"{PWD}/config.json", "r") as config_file:
        config = json.load(config_file)

device_config = config["Device"]
Model = config["Model"]


def setup():
    globals
    global device
    if device_config == "ask":
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print('"{1}" found Microphone with device_index={0})'.format(index, name))

        device1 = input(f"Please choose your Microphone with device_index=")
        device = int(device1)
        print(f"You have selected device_index={device1}")
    else:
        device = int(device_config)
    if Model == "ask":
        questions = [
            inquirer.List(
                "model",
                message="Please Choose your Whisper Model",
                choices=[
                    "tiny.en",
                    "tiny",
                    "base.en",
                    "base",
                    "small.en",
                    "small",
                    "medium.en",
                    "medium",
                    "large",
                    "large-v1",
                    "large-v2",
                    "large-v3",
                    "Custom",
                ],
            ),
        ]
        answers = inquirer.prompt(questions)
        global whisper_model
        whisper_model = answers["model"]
        if whisper_model == "Custom":
            whisper_model = input("Please type in your whisper model : ")
    else:
        whisper_model = Model
    print(f"Selected Whisper Model = {whisper_model}")


def main():
    globals

    if os.path.isfile("temp.flac"):
        os.remove("temp.flac")
    print("Start listening to the microphone.")

    # Initialize model once outside the loop
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")

    while True:
        try:
            r = sr.Recognizer()
            with sr.Microphone(device_index=device) as source:
                audio = r.listen(source)

            if audio is None:
                print("No audio detected, retrying...")
                continue

            # write audio to a FLAC file
            with open("temp.flac", "wb") as f:
                f.write(audio.get_flac_data())

            segments, _ = model.transcribe("temp.flac", language="zh")

            for segment in segments:
                print(f"{segment.text}")
                with open("obs.txt", "w", encoding="utf-8") as f:
                    f.write(segment.text)

            # Clean up temp file
            if os.path.exists("temp.flac"):
                os.remove("temp.flac")

        except KeyboardInterrupt:
            print("\nStopping...")
            if os.path.exists("temp.flac"):
                os.remove("temp.flac")
            break

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if os.path.exists("temp.flac"):
                os.remove("temp.flac")
            print("Retrying...")
            time.sleep(1)
            continue


if __name__ == "__main__":
    setup()
    main()
