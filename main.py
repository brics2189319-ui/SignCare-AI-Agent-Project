import customtkinter as ctk
import cv2
import pyttsx3
import os
import threading
import queue
import time


from PIL import Image, ImageTk
from predictor import SignLanguagePredictor
from word_dictionary import WORDS
from datetime import datetime
from deep_translator import GoogleTranslator

# -----------------------------
# Theme
# -----------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SignCareApp(ctk.CTk):



    def __init__(self):
    

        super().__init__()
            
        self.predictor = SignLanguagePredictor()
 
        # =====================================================
    # Speech System
    # =====================================================

        self.speech_queue = queue.Queue()

        self.speech_thread = threading.Thread(
            target=self.speech_worker,
            daemon=True
    )

        self.speech_thread.start()
                
        
        # -----------------------------
        # Window Settings
        # -----------------------------
        self.title("🤟 SignCare AI - Digital Sign Language Translator")
        self.geometry("1600x900")
        self.minsize(1400, 800)

        # Colors
        self.bg_color = "#10141D"
        self.card_color = "#1B2333"
        self.blue = "#3B82F6"
        self.green = "#22C55E"

        self.configure(fg_color=self.bg_color)

        # Variables
        # -----------------------------
        # Prediction Variables
        # -----------------------------

        self.current_letter = "-"
        self.current_word = ""
        self.current_sentence = ""
        self.confidence = 0
        self.suggestions = []
        self.selected_suggestion = ""
        # Sentence Builder
        self.current_sentence = ""
        # -----------------------------
        # Stable Prediction System
        # -----------------------------

        self.last_prediction = None
        self.prediction_count = 0
        self.required_stable_frames = 15

        # Prevent adding same letter repeatedly
        self.last_added_letter = None

        # Camera Variables
        self.cap = None
        self.camera_running = False

        # =====================================================
        # Background AI Prediction
        # =====================================================

        self.latest_frame = None
        self.latest_result = None

        self.prediction_lock = threading.Lock()

        self.prediction_thread = None
        self.prediction_running = False


        # Build UI
        self.create_sidebar()
        self.create_main()
        self.bind("<Key>", self.select_suggestion)
        self.bind("<space>", lambda e: self.add_space())
        self.bind("<BackSpace>", lambda e: self.backspace())

    # =====================================================
    # Sidebar
    # =====================================================

    def create_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self,
            width=220,
            corner_radius=0,
            fg_color="#161B26"
        )

        self.sidebar.pack(side="left", fill="y")

        self.logo = ctk.CTkLabel(
            self.sidebar,
            text="🤟\nSignCare AI",
            font=("Arial", 28, "bold")
        )

        self.logo.pack(pady=30)

        self.start_btn = ctk.CTkButton(
            self.sidebar,
            text="▶ Start Camera",
            height=45,
            command=self.start_camera
        )

        self.start_btn.pack(fill="x", padx=20, pady=10)

        self.stop_btn = ctk.CTkButton(
            self.sidebar,
            text="⏹ Stop Camera",
            height=45,
            command=self.stop_camera
        )

        self.stop_btn.pack(fill="x", padx=20, pady=10)
        self.speak_btn = ctk.CTkButton(
        self.sidebar,
        text="🔊 Speak",
        height=45,
        command=self.speak_word
)

        self.speak_btn.pack(
            fill="x",
            padx=20,
            pady=10
    )

        


        self.save_btn = ctk.CTkButton(
            self.sidebar,
            text="💾 Save History",
            height=45,
            command=self.save_history
        )

        self.save_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )


        self.history_btn = ctk.CTkButton(
            self.sidebar,
            text="📖 View History",
            height=45,
            command=self.view_history
        )

        self.history_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )

        # =====================================================
        # ASL Alphabet Guide Button
        # =====================================================

        self.asl_guide_btn = ctk.CTkButton(
            self.sidebar,
            text="📚 ASL Alphabet",
            height=45,
            command=self.show_asl_guide
        )

        self.asl_guide_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )

        self.favorite_btn = ctk.CTkButton(
        self.sidebar,
        text="⭐ Add Favorite",
        height=45,
        command=self.add_favorite
            )

        self.favorite_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )


        self.view_fav_btn = ctk.CTkButton(
            self.sidebar,
            text="📖 Favorites",
            height=45,
            command=self.view_favorites
        )

        self.view_fav_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )


        self.clear_fav_btn = ctk.CTkButton(
            self.sidebar,
            text="🗑 Clear Favorites",
            height=45,
            fg_color="#DC2626",
            hover_color="#B91C1C",
            command=self.clear_favorites
        )

        self.clear_fav_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )
        
        self.space_btn = ctk.CTkButton(
            self.sidebar,
            text="↵ Add Space",
            height=45,
            command=self.add_space
        )

        self.space_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )


        self.backspace_btn = ctk.CTkButton(
            self.sidebar,
            text="⌫ Backspace",
            height=45,
            command=self.backspace
        )

        self.backspace_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )


        self.clear_sentence_btn = ctk.CTkButton(
            self.sidebar,
            text="🧹 Clear Sentence",
            height=45,
            command=self.clear_sentence
        )

        self.clear_sentence_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )
        
        

        self.exit_btn = ctk.CTkButton(
            self.sidebar,
            text="❌ Exit",
            fg_color="red",
            hover_color="#B91C1C",
            command=self.destroy
        )

        self.exit_btn.pack(side="bottom", fill="x", padx=20, pady=20)

        self.clear_history_btn = ctk.CTkButton(
            self.sidebar,
            text="🗑 Clear History",
            height=45,
            fg_color="#DC2626",
            hover_color="#B91C1C",
            command=self.clear_history
        )

        self.clear_history_btn.pack(
            fill="x",
            padx=20,
            pady=10
)
        
        
        
        
            
        self.delete_letter_btn = ctk.CTkButton(
            self.sidebar,
            text="⌫ Delete Letter",
            height=45,
            fg_color="#F59E0B",
            hover_color="#D97706",
            command=self.delete_last_letter
        )

        self.delete_letter_btn.pack(
            fill="x",
            padx=20,
            pady=10
    )
        
        
        
        
        
        self.translate_btn = ctk.CTkButton(
            self.sidebar,
            text="🌐 Translate",
            height=45,
            command=self.translate_word
        )

        self.translate_btn.pack(
            fill="x",
            padx=20,
            pady=10
        )
    # =====================================================
    # Main Area
    # =====================================================

    def create_main(self):

        self.main = ctk.CTkFrame(
            self,
            fg_color=self.bg_color
        )

        self.main.pack(side="right", fill="both", expand=True)

        self.create_header()
        self.create_dashboard()

    # =====================================================
    # Header
    # =====================================================

    def create_header(self):

        self.header = ctk.CTkFrame(
            self.main,
            height=70,
            fg_color=self.card_color
        )

        self.header.pack(fill="x", padx=15, pady=15)

        self.title_label = ctk.CTkLabel(
            self.header,
            text="Digital Sign Language Translator",
            font=("Arial", 28, "bold")
        )

        self.title_label.pack(side="left", padx=25)

        self.theme_switch = ctk.CTkSwitch(
            self.header,
            text="Dark Mode"
        )

        self.theme_switch.pack(side="right", padx=25)
           
    # =====================================================
    # Dashboard
        # =====================================================
    def create_dashboard(self):

        self.dashboard = ctk.CTkFrame(
            self.main,
            fg_color=self.bg_color
        )

        self.dashboard.pack(
            fill="both",
            expand=True,
            padx=15
        )

        self.create_camera_panel()

        self.right_panel = ctk.CTkFrame(
            self.dashboard,
            width=400,
            fg_color=self.bg_color
        )

        self.right_panel.pack(
            side="right",
            fill="y",
            padx=(10, 0)
        )

        self.right_panel.pack_propagate(False)

        self.create_alphabet_card()
        self.create_word_card()
        self.create_sentence_card()
        self.create_suggestion_card()
        self.create_confidence_card()
        self.create_status_card()
            
    # =====================================================
    # Camera
    # =====================================================

    def create_camera_panel(self):

        self.camera_panel = ctk.CTkFrame(
            self.dashboard,
            fg_color=self.card_color,
            corner_radius=15
        )

        self.camera_panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        title = ctk.CTkLabel(
            self.camera_panel,
            text="📷 Live Camera",
            font=("Arial", 22, "bold")
        )

        title.pack(pady=15)

        self.camera_label = ctk.CTkLabel(
            self.camera_panel,
            text="Camera Preview",
            width=750,
            height=550,
            fg_color="#0F172A",
            corner_radius=15
        )

        self.camera_label.pack(padx=20, pady=20)

    # =====================================================
    # Alphabet Card
    # =====================================================

    def create_alphabet_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color
        )

        card.pack(fill="x", pady=8)

        ctk.CTkLabel(
            card,
            text="Detected Alphabet",
            font=("Arial", 18, "bold")
        ).pack(pady=(15, 5))

        self.alphabet_label = ctk.CTkLabel(
            card,
            text="-",
            font=("Arial", 60, "bold"),
            text_color=self.blue
        )

        self.alphabet_label.pack(pady=10)
        
    # =====================================================
    # Current Word Card
    # =====================================================

    def create_word_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color,
            corner_radius=12
        )

        card.pack(
            fill="x",
            pady=8
        )

        title = ctk.CTkLabel(
            card,
            text="Current Word",
            font=("Arial", 18, "bold")
        )

        title.pack(
            pady=(15, 5)
        )

        self.word_label = ctk.CTkLabel(
            card,
            text="Waiting...",
            font=("Arial", 28, "bold"),
            text_color=self.green
        )

        self.word_label.pack(
            pady=10
        )

    # =====================================================
    # Word Card
    # =====================================================

    def create_sentence_card(self):

        self.sentence_card = ctk.CTkFrame(
            self.right_panel,
            corner_radius=12,
            fg_color=self.card_color
        )

        self.sentence_card.pack(
            fill="x",
            pady=8
        )

        title = ctk.CTkLabel(
            self.sentence_card,
            text="📝 Current Sentence",
            font=("Arial", 18, "bold")
        )

        title.pack(
            pady=(15, 8)
        )

        self.sentence_label = ctk.CTkLabel(
            self.sentence_card,
            text="Waiting...",
            font=("Arial", 20, "bold"),
            text_color="#FFD700",
            wraplength=350
        )

        self.sentence_label.pack(
            fill="x",
            padx=15,
            pady=(5, 15)
        )
        
        
    # =====================================================
    # AI Word Suggestions
    # =====================================================

    
# =====================================================
# Select AI Suggestion
# =====================================================

    def select_suggestion(self, index):

        if not hasattr(self, "suggestions"):
            return

        if index >= len(self.suggestions):
            return

        selected_word = self.suggestions[index]

        if not selected_word:
            return

        self.current_word = selected_word

        self.word_label.configure(
            text=selected_word
        )

        self.suggestion_status.configure(
            text=f"Selected: {selected_word}"
        )

        print(
            f"Suggestion selected: {selected_word}"
        )
    # =====================================================
    # Sentence Card
    # =====================================================

    def create_sentence_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color,
            corner_radius=12
        )

        card.pack(
            fill="x",
            pady=8
        )

        ctk.CTkLabel(
            card,
            text="📝 Sentence",
            font=("Arial", 18, "bold")
        ).pack(
            pady=(15, 5)
        )

        self.sentence_label = ctk.CTkLabel(
            card,
            text="Waiting...",
            font=("Arial", 20, "bold"),
            text_color=self.green,
            wraplength=330
        )

        self.sentence_label.pack(
            fill="x",
            padx=15,
            pady=(5, 15)
        )
            # =====================================================
    # Update AI Suggestions
    # =====================================================

    def update_suggestions(self):

        self.suggestions = []

        if self.current_word.strip() == "":
            self.suggestion_status.configure(text="Waiting...")
            return

        prefix = self.current_word.upper()

        for word in WORDS:

            if word.startswith(prefix):
                self.suggestions.append(word)

            if len(self.suggestions) == 5:
                break

        if self.suggestions:

            text = ""

            for i, word in enumerate(self.suggestions, start=1):
                text += f"{i}. {word}\n"

            self.suggestion_status.configure(text=text)

        else:

            self.suggestion_status.configure(
                text="No Suggestions"
            )
        
   # =====================================================
# Accept AI Suggestion
# =====================================================

    def accept_suggestion(self, index):

        if index < len(self.suggestions):

            self.current_word = self.suggestions[index]

            self.word_label.configure(
                text=self.current_word
            )

            self.suggestion_status.configure(
                text="✔ Selected\n" + self.current_word
            )

            self.model_status.configure(
                text="🟢 AI Suggestion Accepted"
            )

            # Speak completed word
            self.speak_text_async(
                self.current_word
            )

            # Save automatically
            self.save_history()

            # Add space for next word
            self.current_word += " "
            
            
        # =====================================================
# Select Suggestion
# =====================================================

    def select_suggestion(self, event):

        if event.char in ["1", "2", "3", "4", "5"]:

            index = int(event.char) - 1

            if index < len(self.suggestions):

                self.current_word = self.suggestions[index]

                self.word_label.configure(
                    text=self.current_word
                )

                self.suggestion_status.configure(
                    text="✔ Selected:\n" + self.current_word
                )

                self.model_status.configure(
                    text="🟢 AI Suggestion Selected"
                )
            
        
 
    # =====================================================
    # AI Word Suggestion Card
    # =====================================================

    def create_suggestion_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color,
            corner_radius=12
        )

        card.pack(
            fill="x",
            pady=8
        )

        ctk.CTkLabel(
            card,
            text="🤖 AI Word Suggestions",
            font=("Arial", 18, "bold")
        ).pack(
            pady=(15, 10)
        )

        self.suggestion_buttons = []

        for i in range(3):

            btn = ctk.CTkButton(
                card,
                text=f"{i + 1}. Waiting...",
                height=40,
                command=lambda index=i: self.accept_suggestion(index)
            )

            btn.pack(
                fill="x",
                padx=15,
                pady=5
            )

            self.suggestion_buttons.append(btn)

        self.suggestion_status = ctk.CTkLabel(
            card,
            text="Start signing to get suggestions",
            font=("Arial", 13),
            text_color="#AAB4C3"
        )

        self.suggestion_status.pack(
            pady=(5, 15)
        )
    # =====================================================
    # Status Card
    # =====================================================

    def create_status_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color
        )

        card.pack(fill="x", pady=8)

        ctk.CTkLabel(
            card,
            text="AI Status",
            font=("Arial", 18, "bold")
        ).pack(pady=(15, 10))

        self.camera_status = ctk.CTkLabel(
            card,
            text="🔴 Camera : Offline",
            anchor="w"
        )

        self.camera_status.pack(fill="x", padx=20)

        self.model_status = ctk.CTkLabel(
            card,
            text="🟢 Model : Ready",
            anchor="w"
        )

        self.model_status.pack(fill="x", padx=20)

        self.hand_status = ctk.CTkLabel(
            card,
            text="🟠 Hand : Not Detected",
            anchor="w"
        )

        self.hand_status.pack(fill="x", padx=20, pady=(0, 15))

    # =====================================================
    # Confidence Card
    # =====================================================

    def create_confidence_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color,
            corner_radius=12
        )

        card.pack(
            fill="x",
            pady=8
        )

        ctk.CTkLabel(
            card,
            text="Prediction Confidence",
            font=("Arial", 18, "bold")
        ).pack(
            pady=(15, 10)
        )

        self.progress = ctk.CTkProgressBar(
            card
        )

        self.progress.pack(
            fill="x",
            padx=20
        )

        self.progress.set(0)

        self.confidence_label = ctk.CTkLabel(
            card,
            text="0%",
            font=("Arial", 15)
        )

        self.confidence_label.pack(
            pady=10
        )

    # =====================================================
    # Start Camera
    # =====================================================

    def start_camera(self):

        if self.camera_running:
            return

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            self.model_status.configure(
                text="🔴 Camera Failed"
            )

            return

        # Reduce camera resolution
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.camera_running = True

        self.prediction_running = True

        self.camera_status.configure(
            text="🟢 Camera : Online"
        )

        self.model_status.configure(
            text="🟢 AI : Running"
        )

        # Start AI prediction thread

        self.prediction_thread = threading.Thread(
            target=self.prediction_worker,
            daemon=True
        )

        self.prediction_thread.start()

        # Start camera display

        self.update_camera()
    # =====================================================
    # Stop Camera
    # =====================================================

    def stop_camera(self):

        self.camera_running = False

        self.prediction_running = False

        # Clear AI data

        with self.prediction_lock:

            self.latest_frame = None

            self.latest_result = None

        # Release camera

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        # Reset UI

        self.camera_label.configure(
            image=None,
            text="Camera Preview"
        )

        self.camera_status.configure(
            text="🔴 Camera : Offline"
        )

        self.model_status.configure(
            text="🟢 Model : Ready"
        )

        self.alphabet_label.configure(
            text="-"
        )

        self.progress.set(0)

        self.confidence_label.configure(
            text="0%"
        )

        self.hand_status.configure(
            text="🟠 Hand : Not Detected"
        )


    # =====================================================
    # Speak Current Word - Non Blocking
    # =====================================================
    # # =====================================================
    # Speak Current Sentence
    # =====================================================

    def speak_word(self):

        try:

            # Get complete sentence
            sentence = self.current_sentence.strip()

            # If sentence is empty, use current word
            if not sentence:

                sentence = self.current_word.strip()

            if not sentence:

                self.model_status.configure(
                    text="🟡 Nothing to speak"
                )

                return

            print("Speech requested:", sentence)

            # Put speech request into queue
            self.speech_queue.put(sentence)

            self.model_status.configure(
                text="🔊 Speaking..."
            )

        except Exception as error:

            print("Speak Button Error:", error)

            self.model_status.configure(
                text="🔴 Speech Error"
            )
            
        
    # =====================================================
    # Speech Worker
    # =====================================================

    def speech_worker(self):

        try:
            # Create ONE engine for the worker thread
            engine = pyttsx3.init()

            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)

            while True:

                text = self.speech_queue.get()

                try:

                    # Stop worker
                    if text is None:
                        break

                    text = str(text).strip()

                    if not text:
                        continue

                    print("🔊 Speaking:", text)

                    # Speak
                    engine.say(text)
                    engine.runAndWait()

                    print("✅ Speech completed")

                    # Update UI safely
                    self.after(
                        0,
                        lambda: self.model_status.configure(
                            text="🟢 Speech Completed"
                        )
                    )

                except Exception as error:

                    print("Speech Error:", error)

                    self.after(
                        0,
                        lambda: self.model_status.configure(
                            text="🔴 Speech Error"
                        )
                    )

                finally:
                    self.speech_queue.task_done()

            # Close engine only when application exits
            engine.stop()

        except Exception as error:

            print("Speech Worker Error:", error)
        
        
    # =====================================================
    # Async Speech
    # =====================================================

    def speak_text_async(self, text):

        text = text.strip()

        if not text:
            return

        self.speech_queue.put(text)
    
    # =====================================================
    # CLEAR CURRENT WORD ONLY
    # =====================================================

    def clear_word(self):

        # Clear ONLY the current word
        self.current_word = ""

        self.current_letter = "-"

        self.last_prediction = None
        self.prediction_count = 0
        self.last_added_letter = None

        # Reset current word display
        self.word_label.configure(
            text="Waiting..."
        )

        # IMPORTANT:
        # Do NOT touch self.current_sentence

        # Reset alphabet
        self.alphabet_label.configure(
            text="-"
        )

        # Reset confidence
        self.progress.set(0)

        self.confidence_label.configure(
            text="0%"
        )

        # Reset suggestions
        self.suggestions = []

        self.suggestion_status.configure(
            text="Start signing..."
        )

        self.model_status.configure(
            text="🟢 Current Word Cleared"
        )
        
    # =====================================================
    # Update AI Word Suggestions
    # =====================================================

    def update_word_suggestions(self):

        prefix = self.current_word.strip().upper()

        # No word yet
        if not prefix:

            self.suggestions = []

            for i, btn in enumerate(self.suggestion_buttons):

                btn.configure(
                    text=f"{i + 1}. Waiting..."
                )

            self.suggestion_status.configure(
                text="Start signing to get suggestions"
            )

            return

        # Find matching words
        matches = []

        for word in WORDS:

            word = str(word).upper()

            if word.startswith(prefix):

                matches.append(word)

        # Maximum 3 suggestions
        matches = matches[:3]

        self.suggestions = matches

        # Update suggestion buttons
        for i in range(3):

            if i < len(matches):

                self.suggestion_buttons[i].configure(
                    text=f"{i + 1}. {matches[i]}"
                )

            else:

                self.suggestion_buttons[i].configure(
                    text=f"{i + 1}. ---"
                )

        # Update status
        if matches:

            self.suggestion_status.configure(
                text=f"{len(matches)} suggestion(s) found"
            )

        else:

            self.suggestion_status.configure(
                text="No suggestions found"
            )
            
    # =====================================================
    # CLEAR SUGGESTION BUTTONS
    # =====================================================

    def clear_suggestion_buttons(self):

        if hasattr(self, "suggestion_buttons"):

            for button in self.suggestion_buttons:

                try:
                    button.destroy()
                except:
                    pass

            self.suggestion_buttons = []
            
                
    # =====================================================
    # SELECT WORD SUGGESTION
    # =====================================================

    def select_word_suggestion(self, word):

        # Replace current partial word
        self.current_word = word

        # Update current word display
        self.word_label.configure(
            text=self.current_word
        )

        # Clear suggestions
        self.clear_suggestion_buttons()

        # Show message
        self.suggestion_status.configure(
            text=f"✓ {word}"
        )

        print(
            f"Selected suggestion: {word}"
        )
            
            
    # =====================================================
    # AI WORD SUGGESTIONS CARD
    # =====================================================

    def create_word_suggestion_card(self):

        card = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.card_color
        )

        card.pack(
            fill="x",
            pady=8
        )

        ctk.CTkLabel(
            card,
            text="🤖 AI Word Suggestions",
            font=("Arial", 18, "bold")
        ).pack(
            pady=(15, 5)
        )

        self.suggestion_label = ctk.CTkLabel(
            card,
            text="Waiting...",
            font=("Arial", 18),
            justify="left"
        )

        self.suggestion_label.pack(
            padx=20,
            pady=15,
            anchor="w"
        )
        # =====================================================
        #: Add Space Function
        # =====================================================
    # =====================================================
    # ADD SPACE / COMPLETE CURRENT WORD
    # =====================================================

    def add_space(self):

        # Get current word
        word = self.current_word.strip()

        # Nothing to add
        if not word:
            self.model_status.configure(
                text="🟠 No current word"
            )
            return

        # Add word to sentence
        if self.current_sentence.strip():

            self.current_sentence += " " + word

        else:

            self.current_sentence = word

        # IMPORTANT:
        # Keep the sentence separately.
        # Only current_word is cleared.
        self.current_word = ""

        # Show new/current word as empty
        self.word_label.configure(
            text="Waiting..."
        )

        # KEEP OLD WORD IN SENTENCE
        self.sentence_label.configure(
            text=self.current_sentence
        )

        # Reset detected alphabet
        self.current_letter = "-"

        self.alphabet_label.configure(
            text="-"
        )

        # Reset confidence
        self.confidence = 0

        self.progress.set(0)

        self.confidence_label.configure(
            text="0%"
        )

        # Reset prediction stability
        self.last_prediction = None
        self.prediction_count = 0
        self.last_added_letter = None

        # Clear suggestions for next word
        self.suggestions = []

        self.suggestion_status.configure(
            text="Start next word..."
        )

        # Status
        self.model_status.configure(
            text=f"🟢 Added: {word}"
        )

        print(
            "Current Word:",
            self.current_word
        )

        print(
            "Current Sentence:",
            self.current_sentence
        )
    # =====================================================
    # Backspace Function
    # =====================================================      
        
        
    def backspace(self):

        if len(self.current_word) > 0:

            self.current_word = self.current_word[:-1]

            if self.current_word == "":

                self.word_label.configure(
                    text="Waiting..."
                )

            else:

                self.word_label.configure(
                    text=self.current_word
                )
                
                
       
         # : Clear Sentence
                   
    # =====================================================
    # CLEAR COMPLETE SENTENCE
    # =====================================================

    def clear_sentence(self):

        self.current_sentence = ""
        self.current_word = ""

        self.current_letter = "-"

        self.last_prediction = None
        self.prediction_count = 0
        self.last_added_letter = None

        # Current word
        self.word_label.configure(
            text="Waiting..."
        )

        # Complete sentence
        self.sentence_label.configure(
            text="Waiting..."
        )

        # Alphabet
        self.alphabet_label.configure(
            text="-"
        )

        # Confidence
        self.progress.set(0)

        self.confidence_label.configure(
            text="0%"
        )

        # Suggestions
        self.suggestions = []

        self.suggestion_status.configure(
            text="Start signing..."
        )

        self.model_status.configure(
            text="🟢 Sentence Cleared"
        )
            # =====================================================
    # Delete Last Letter
    # =====================================================

    def delete_last_letter(self):

        if len(self.current_word) > 0:

            self.current_word = self.current_word[:-1]

            if self.current_word == "":
                self.word_label.configure(
                    text="Waiting..."
                )
            else:
                self.word_label.configure(
                    text=self.current_word
                )

            self.model_status.configure(
                text="🟠 Last Letter Deleted"
            )

            print("Last letter removed.")        
            
        
        

        
        
            # =====================================================
    # Save Translation History
    # =====================================================

    def save_history(self):

            # Check if word is empty
            if self.current_word.strip() == "":
                self.model_status.configure(
                    text="🟠 Nothing to Save"
                )
                print("No word available to save.")
                return

            # Create history folder
            os.makedirs(
                "history",
                exist_ok=True
            )

            # Create date and time
            current_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # File path
            history_path = os.path.join(
                "history",
                "history.txt"
            )

            # Save word
            with open(
                history_path,
                "a",
                encoding="utf-8"
            ) as file:

                file.write(
                    f"[{current_time}] "
                    f"{self.current_word}\n"
                )

            # Update UI
            self.model_status.configure(
                text="🟢 History Saved"
            )

            print(
                f"Saved: {self.current_word}"
            )
            
             # =====================================================
             # favouritr function
             # =====================================================
                
    def add_favorite(self):

        if self.current_word.strip() == "":
            self.model_status.configure(
                text="🟠 No Word"
            )
            return

        os.makedirs(
            "favorites",
            exist_ok=True
        )

        file_path = os.path.join(
            "favorites",
            "favorites.txt"
        )

        with open(
            file_path,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                self.current_word + "\n"
            )

        self.model_status.configure(
            text="⭐ Favorite Saved"
        )     
                
            
            
            # =====================================================
             # view  favouritr function
             # =====================================================
            
    def view_favorites(self):

        path = os.path.join(
            "favorites",
            "favorites.txt"
        )

        win = ctk.CTkToplevel(self)

        win.title("Favorite Words")

        win.geometry("600x500")

        textbox = ctk.CTkTextbox(
            win,
            width=560,
            height=420
        )

        textbox.pack(
            padx=20,
            pady=20,
            fill="both",
            expand=True
        )

        if os.path.exists(path):

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                data = file.read()

            if data.strip() == "":
                data = "No Favorites."

        else:

            data = "No Favorites."

        textbox.insert("1.0", data)

        textbox.configure(
            state="disabled"
        )

        win.focus()
            
            # =====================================================
             # clear  favouritr function
             # =====================================================            
    def clear_favorites(self):

        path = os.path.join(
            "favorites",
            "favorites.txt"
        )

        if os.path.exists(path):

            open(path, "w").close()

        self.model_status.configure(
            text="🗑 Favorites Cleared"
        )

            
            # =====================================================
            # View Translation History
            # =====================================================

    def view_history(self):

        history_path = os.path.join(
            "history",
            "history.txt"
        )

        history_window = ctk.CTkToplevel(self)
        history_window.title("Translation History")
        history_window.geometry("700x500")

        title = ctk.CTkLabel(
            history_window,
            text="📜 Translation History",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=15)

        textbox = ctk.CTkTextbox(
            history_window,
            width=650,
            height=350
        )
        textbox.pack(
            padx=20,
            pady=10,
            fill="both",
            expand=True
        )

        if os.path.exists(history_path):

            with open(
                history_path,
                "r",
                encoding="utf-8"
            ) as file:

                history = file.read()

            if history.strip() == "":
                history = "No translation history available."

        else:

            history = "History file not found."

        textbox.insert("1.0", history)
        textbox.configure(state="disabled")

        close_btn = ctk.CTkButton(
            history_window,
            text="Close",
            command=history_window.destroy
        )

        close_btn.pack(pady=15)
   
   
    # =====================================================
    # ASL Alphabet Guide
    # =====================================================

    def show_asl_guide(self):

        guide_window = ctk.CTkToplevel(self)

        guide_window.title("ASL Alphabet Guide")
        guide_window.geometry("850x900")

        guide_window.transient(self)

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        title = ctk.CTkLabel(
            guide_window,
            text="🤟 American Sign Language Alphabet",
            font=("Arial", 26, "bold")
        )

        title.pack(
            pady=(15, 5)
        )

        subtitle = ctk.CTkLabel(
            guide_window,
            text="Use this guide to learn the hand signs for A-Z",
            font=("Arial", 14)
        )

        subtitle.pack(
            pady=(0, 10)
        )

        # -------------------------------------------------
        # Scrollable frame
        # -------------------------------------------------

        scroll_frame = ctk.CTkScrollableFrame(
            guide_window,
            width=780,
            height=750
        )

        scroll_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

        image_path = os.path.join(
            "assets",
            "asl_alphabet.jpeg"
        )

        if os.path.exists(image_path):

            original_image = Image.open(
                image_path
            )

            # Resize image while maintaining aspect ratio
            max_width = 720

            original_width, original_height = original_image.size

            scale = max_width / original_width

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            resized_image = original_image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

            self.asl_guide_image = ImageTk.PhotoImage(
                resized_image
            )

            image_label = ctk.CTkLabel(
                scroll_frame,
                text="",
                image=self.asl_guide_image
            )

            image_label.pack(
                padx=10,
                pady=10
            )

        else:

            error_label = ctk.CTkLabel(
                scroll_frame,
                text=(
                    "ASL alphabet image not found.\n\n"
                    "Please place the image here:\n"
                    "assets/asl_alphabet.jpg"
                ),
                font=("Arial", 16),
                text_color="red"
            )

            error_label.pack(
                pady=100
            )

        # -------------------------------------------------
        # Close button
        # -------------------------------------------------

        close_button = ctk.CTkButton(
            guide_window,
            text="✖ Close",
            height=40,
            command=guide_window.destroy
        )

        close_button.pack(
            pady=15
        )   

    # =====================================================
# Translate Word
# =====================================================

    def translate_word(self):

        word = self.current_word.strip()

        if word == "":

            self.model_status.configure(
                text="🟠 No Word To Translate"
            )

            return

        try:

            hindi = GoogleTranslator(
                source="auto",
                target="hi"
            ).translate(word)

            gujarati = GoogleTranslator(
                source="auto",
                target="gu"
            ).translate(word)

            window = ctk.CTkToplevel(self)

            window.title("🌐 Translation")

            window.geometry("500x350")

            title = ctk.CTkLabel(
                window,
                text="Language Translation",
                font=("Arial",22,"bold")
            )

            title.pack(pady=15)

            english_label = ctk.CTkLabel(
                window,
                text=f"🇬🇧 English:\n{word}",
                font=("Arial",18)
            )

            english_label.pack(pady=10)

            hindi_label = ctk.CTkLabel(
                window,
                text=f"🇮🇳 Hindi:\n{hindi}",
                font=("Arial",18)
            )

            hindi_label.pack(pady=10)

            gujarati_label = ctk.CTkLabel(
                window,
                text=f"🇮🇳 Gujarati:\n{gujarati}",
                font=("Arial",18)
            )

            gujarati_label.pack(pady=10)

            self.model_status.configure(
                text="🟢 Translation Complete"
            )

        except Exception as e:

            self.model_status.configure(
                text="🔴 Translation Failed"
            )

            print(e)
            
    
    # =====================================================
# Delete Translation History
# =====================================================

    def delete_history(self, history_window):

        history_path = os.path.join(
            "history",
            "history.txt"
        )

        # Delete history file
        if os.path.exists(history_path):

            os.remove(history_path)

            # Update main window status
            self.model_status.configure(
                text="🟢 History Deleted"
            )

            # Close history window
            history_window.destroy()

        print("Translation history deleted successfully.")
 
    # =====================================================
# Delete History Button
# =====================================================

        delete_btn = ctk.CTkButton(
            history_window,
            text="🗑 Delete All History",
            height=40,
            fg_color="#DC2626",
            hover_color="#B91C1C",
            command=lambda: self.delete_history(
                history_window
            )
        )

        delete_btn.pack(
            pady=15
)
        
        
        
        
        # =====================================================
        # Clear Translation History
        # =====================================================

    def clear_history(self):

        history_path = os.path.join(
            "history",
            "history.txt"
        )

        try:

            if os.path.exists(history_path):

                with open(
                    history_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    file.write("")

                self.model_status.configure(
                    text="🟢 History Cleared"
                )

                print(
                    "History cleared successfully."
                )

            else:

                self.model_status.configure(
                    text="🟠 No History Found"
                )

                print(
                    "History file not found."
                )

        except Exception as error:

            self.model_status.configure(
                text="🔴 Error Clearing History"
            )

            print(
                "Error:",
                error
        )
            
# =====================================================
# AI Prediction Worker
# =====================================================

    def prediction_worker(self):

        while self.prediction_running:

            frame = None

            # Get latest frame
            with self.prediction_lock:

                if self.latest_frame is not None:

                    frame = self.latest_frame.copy()

                    self.latest_frame = None

            if frame is None:

                time.sleep(0.01)

                continue

            try:

                # AI prediction
                letter, confidence, results = (
                    self.predictor.predict(frame)
                )

                # Store result
                with self.prediction_lock:

                    self.latest_result = (
                        letter,
                        confidence,
                        results
                    )

            except Exception as error:

                print(
                    "Prediction Worker Error:",
                    error
                )

                with self.prediction_lock:

                    self.latest_result = None

                time.sleep(0.05)
            

    # =====================================================
    # Update Camera
    # =====================================================

    def update_camera(self):

        if not self.camera_running:
            return

        try:

            # ---------------------------------
            # Read camera frame
            # ---------------------------------

            ret, frame = self.cap.read()

            if not ret:

                self.after(
                    30,
                    self.update_camera
                )

                return

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            # ---------------------------------
            # Send latest frame to AI worker
            # ---------------------------------

            with self.prediction_lock:

                self.latest_frame = frame.copy()

            # ---------------------------------
            # Get latest AI result
            # ---------------------------------

            result = None

            with self.prediction_lock:

                if self.latest_result is not None:

                    result = self.latest_result

                    self.latest_result = None

            # ---------------------------------
            # Process AI result
            # ---------------------------------

            if result is not None:

                letter, confidence, results = result

                # Draw landmarks
                try:

                    frame = self.predictor.draw_landmarks(
                        frame,
                        results
                    )

                except Exception as error:

                    print(
                        "Landmark Error:",
                        error
                    )

                # ---------------------------------
                # Hand detected
                # ---------------------------------

                if letter is not None:

                    letter = str(letter)

                    self.current_letter = letter

                    self.confidence = confidence

                    # Update alphabet

                    self.alphabet_label.configure(
                        text=letter
                    )

                    # Confidence

                    self.progress.set(
                        confidence
                    )

                    self.confidence_label.configure(
                        text=f"{confidence * 100:.1f}%"
                    )

                    self.hand_status.configure(
                        text="🟢 Hand : Detected"
                    )

                    # ---------------------------------
                    # Stable prediction
                    # ---------------------------------

                    if letter == self.last_prediction:

                        self.prediction_count += 1

                    else:

                        self.last_prediction = letter

                        self.prediction_count = 1

                        self.last_added_letter = None

                    # ---------------------------------
                    # Add letter
                    # ---------------------------------

                    if (
                        self.prediction_count >=
                        self.required_stable_frames
                        and
                        self.last_added_letter != letter
                    ):

                        self.current_word += letter

                        self.last_added_letter = letter

                        self.word_label.configure(
                            text=self.current_word
                        )
                        self.update_word_suggestions()

                        # Update AI suggestions
                        self.update_word_suggestions()

                        # Update suggestions ONLY
                        # when a new letter is added

                        self.update_word_suggestions()

                    # ---------------------------------
                    # Draw prediction
                    # ---------------------------------

                    prediction_text = (
                        f"{letter} "
                        f"({confidence * 100:.1f}%)"
                    )

                    cv2.putText(
                        frame,
                        prediction_text,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )

                else:

                    self.current_letter = "-"

                    self.confidence = 0

                    self.alphabet_label.configure(
                        text="-"
                    )

                    self.progress.set(0)

                    self.confidence_label.configure(
                        text="0%"
                    )

                    self.hand_status.configure(
                        text="🟠 Hand : Not Detected"
                    )

                    self.last_prediction = None

                    self.prediction_count = 0

                    self.last_added_letter = None

            # ---------------------------------
            # Convert frame for Tkinter
            # ---------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                frame_rgb
            )

            image = image.resize(
                (760, 560)
            )

            photo = ImageTk.PhotoImage(
                image
            )

            self.camera_label.configure(
                image=photo,
                text=""
            )

            self.camera_label.image = photo

        except Exception as error:

            print(
                "Camera Error:",
                error
            )

        # ---------------------------------
        # Continue camera loop
        # ---------------------------------

        if self.camera_running:

            self.after(
                30,
                self.update_camera
            )
        
# END OF CLASS


if __name__ == "__main__":
    app = SignCareApp()
    app.mainloop()