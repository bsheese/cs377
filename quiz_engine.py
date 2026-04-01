import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
import random

class QuizApp:
    def __init__(self, data, title="Practice Quiz"):
        self.data = data
        self.title = title
        
        # Shuffle questions and options
        random.shuffle(self.data)
        self.shuffled_options = [random.sample(q['options'], len(q['options'])) for q in self.data]
        
        self.index = 0
        self.user_answers = [None] * len(self.data)
        
        # UI Elements
        self.header = widgets.HTML(f"<h2>{self.title}</h2>")
        self.progress = widgets.HTML()
        
        # Use Output widgets for Math/Markdown support
        self.question_output = widgets.Output()
        self.feedback_output = widgets.Output()
        
        self.options = widgets.RadioButtons(layout={'width': 'max-content'})
        
        # Buttons
        self.btn_back = widgets.Button(description="Back", button_style='warning', icon='arrow-left')
        self.btn_submit = widgets.Button(description="Submit", button_style='primary', icon='check')
        self.btn_next = widgets.Button(description="Next", button_style='info', icon='arrow-right')
        
        # Event Handlers
        self.btn_submit.on_click(self.on_submit)
        self.btn_next.on_click(self.on_next)
        self.btn_back.on_click(self.on_back)
        
        # Layout
        self.controls = widgets.HBox([self.btn_back, self.btn_submit, self.btn_next])
        self.ui = widgets.VBox([
            self.header, 
            self.progress, 
            self.question_output,  # Markdown/Math will render here
            self.options, 
            self.controls, 
            self.feedback_output   # Markdown/Math will render here
        ])
        
        self.update_ui()

    def update_ui(self):
        if self.index < len(self.data):
            q = self.data[self.index]
            
            # Update Progress and Options
            self.progress.value = f"<b>Question {self.index + 1} of {len(self.data)}</b>"
            self.options.options = self.shuffled_options[self.index]
            
            # Render Question Text with Math support
            with self.question_output:
                clear_output(wait=True)
                display(Markdown(q['question']))
            
            # Logic for answered vs unanswered
            if self.user_answers[self.index] is not None:
                self.options.value = self.user_answers[self.index]
                self.options.disabled = True
                self.btn_submit.disabled = True
                self.btn_next.disabled = False
                self.show_feedback_message(self.user_answers[self.index])
            else:
                self.options.value = None
                self.options.disabled = False
                self.btn_submit.disabled = False
                self.btn_next.disabled = True
                with self.feedback_output:
                    clear_output()

            self.btn_back.disabled = (self.index == 0)
        else:
            self.show_results()

    def show_feedback_message(self, selected):
        correct = self.data[self.index]['answer']
        with self.feedback_output:
            clear_output(wait=True)
            if selected == correct:
                display(Markdown("---"))
                display(Markdown("> **✅ Correct!**"))
            else:
                display(Markdown("---"))
                display(Markdown(f"> **❌ Incorrect.**  \n> Correct answer: **{correct}**"))

    def on_submit(self, b):
        if self.options.value is None:
            with self.feedback_output:
                clear_output()
                display(Markdown("<span style='color:red;'>⚠️ Please select an option!</span>"))
            return
        self.user_answers[self.index] = self.options.value
        self.update_ui()

    def on_next(self, b):
        self.index += 1
        self.update_ui()

    def on_back(self, b):
        self.index -= 1
        self.update_ui()

    def show_results(self):
        score = sum(1 for i, ans in enumerate(self.user_answers) if ans == self.data[i]['answer'])
        pct = (score / len(self.data)) * 100
        
        with self.question_output:
            clear_output()
        with self.feedback_output:
            clear_output()
            
        self.ui.children = [widgets.HTML(f"""
            <div style="text-align: center; padding: 20px;">
                <h2>Quiz Finished!</h2>
                <h1 style="font-size: 50px;">{pct:.1f}%</h1>
                <p style="font-size: 18px;">You scored <b>{score}</b> out of <b>{len(self.data)}</b></p>
            </div>
        """)]
