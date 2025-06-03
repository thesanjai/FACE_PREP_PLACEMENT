import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say 'exit' to terminate the program.")
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready to receive your voice input.")
        
        while True:
            print("\nPlease speak something...")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print("You said:", text)
                if text.strip().lower() == "exit":
                    print("Exiting...")
                    break
            except sr.WaitTimeoutError:
                print("No speech detected. Try again.")
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
            except sr.RequestError:
                print("Sorry, could not request results from the speech recognition service.")

if __name__ == "__main__":
    speech_to_text()
