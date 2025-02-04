import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import speech_recognition as sr
import tempfile
import random

class LearningAssistant:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.recognizer = sr.Recognizer()

        # Define code analysis tasks
        self.code_analysis_tasks = {
            "explain": "Explain what this code does in detail",
            "debug": "Find and explain potential bugs or issues in this code",
            "review": "Review this code for best practices and improvements",
            "analyze": "Analyze this code for complexity and potential issues",
            "optimize": "Suggest optimizations for this code",
            "refactor": "Suggest how this code could be refactored",
            "secure": "Analyze this code for security vulnerabilities",
            "complexity": "Analyze the time and space complexity of this code"
        }

        self.supported_languages = ["python", "C++", "java", "C", "JavaScript"]
        
        self.random_topics = [
            "Introduction to Machine Learning",
            "The Future of Quantum Computing",
            "How Blockchain Technology Works",
            "Understanding Big Data",
            "The Role of Artificial Intelligence",
            "Cybersecurity Basics",
            "Evolution of Programming Languages",
            "Cloud Computing Fundamentals",
            "Impact of IoT",
            "AI Ethics"
        ]

        self.dsa_topics = [
            "Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Graphs",
            "Sorting", "Searching", "Dynamic Programming", "Greedy Algorithms",
            "Hash Tables", "Recursion", "Binary Search", "Heap", "Trie"
        ]

        self.difficulty_levels = ["easy", "medium", "hard"]

    def create_prompt(self, user_input):
        return f"<s>[INST] {user_input} [/INST]"

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response by removing prompt and instruction tokens
        response = full_response.split("[/INST]")[-1].strip()
        response = response.replace("<s>", "").replace("</s>", "").strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        
        return response

    def text_to_speech(self, text):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            return fp.name

    def generate_blog(self, topic=None):
        if topic is None:
            topic = random.choice(self.random_topics)
        prompt = self.create_prompt(f"Write a detailed blog post about {topic} for computer science students.")
        return topic, self.generate_response(prompt)

    def generate_dsa_question(self, topic, difficulty):
        prompt = self.create_prompt(
            f"""Create a Data Structures and Algorithms question about {topic} with {difficulty} difficulty.
            
            Format:
            Title:
            Difficulty: {difficulty}
            Topic: {topic}
            
            Problem Statement:
            [Problem description]
            
            Input Format:
            [Input format]
            
            Output Format:
            [Output format]
            
            Example:
            Input: [Sample input]
            Output: [Sample output]
            
            Constraints:
            [Constraints]"""
        )
        return self.generate_response(prompt)

    def execute_code(self, code, language):
        if language.lower() not in self.supported_languages:
            return f"Language {language} is not supported"
        
        prompt = self.create_prompt(
            f"""As a {language} interpreter, what would be the output of this code? 
            If there are errors, show only the error message.
            
            {code}"""
        )
        return self.generate_response(prompt)

    def analyze_code(self, code, task):
        if task not in self.code_analysis_tasks:
            return "Invalid analysis task"
            
        prompt = self.create_prompt(
            f"""Analyze this code according to the task: {self.code_analysis_tasks[task]}
            
            Code:
            {code}"""
        )
        return self.generate_response(prompt)

def create_gradio_interface():
    assistant = LearningAssistant()
    
    # Blog Generation Interface
    def blog_interface(topic=None):
        topic, blog = assistant.generate_blog(topic)
        return f"Topic: {topic}\n\n{blog}"
    
    # DSA Question Generation Interface
    def dsa_interface(topic, difficulty):
        return assistant.generate_dsa_question(topic, difficulty)
    
    # Code Analysis Interface
    def code_analysis_interface(code, task):
        return assistant.analyze_code(code, task)
    
    # Code Execution Interface
    def code_execution_interface(code, language):
        return assistant.execute_code(code, language)
    
    # Text to Speech Interface
    def tts_interface(text):
        return assistant.text_to_speech(text)

    # Create the Gradio interface
    with gr.Blocks(title="Learning Assistant") as interface:
        gr.Markdown("# AI Learning Assistant")
        
        with gr.Tab("Blog Generator"):
            blog_input = gr.Textbox(label="Topic (optional)")
            blog_button = gr.Button("Generate Blog")
            blog_output = gr.Textbox(label="Generated Blog", lines=10)
            blog_button.click(blog_interface, inputs=blog_input, outputs=blog_output)

        with gr.Tab("DSA Question Generator"):
            dsa_topic = gr.Dropdown(choices=assistant.dsa_topics, label="Topic")
            dsa_difficulty = gr.Dropdown(choices=assistant.difficulty_levels, label="Difficulty")
            dsa_button = gr.Button("Generate Question")
            dsa_output = gr.Textbox(label="Generated Question", lines=10)
            dsa_button.click(dsa_interface, inputs=[dsa_topic, dsa_difficulty], outputs=dsa_output)

        with gr.Tab("Code Analysis"):
            code_input = gr.Code(label="Code")
            analysis_task = gr.Dropdown(choices=list(assistant.code_analysis_tasks.keys()), label="Analysis Task")
            analysis_button = gr.Button("Analyze Code")
            analysis_output = gr.Textbox(label="Analysis Result", lines=10)
            analysis_button.click(code_analysis_interface, inputs=[code_input, analysis_task], outputs=analysis_output)

        with gr.Tab("Code Execution"):
            exec_code_input = gr.Code(label="Code")
            language_input = gr.Dropdown(choices=assistant.supported_languages, label="Language")
            exec_button = gr.Button("Execute Code")
            exec_output = gr.Textbox(label="Execution Output", lines=5)
            exec_button.click(code_execution_interface, inputs=[exec_code_input, language_input], outputs=exec_output)

        with gr.Tab("Text to Speech"):
            text_input = gr.Textbox(label="Text to Convert")
            tts_button = gr.Button("Convert to Speech")
            audio_output = gr.Audio(label="Generated Speech")
            tts_button.click(tts_interface, inputs=text_input, outputs=audio_output)

    return interface

# Create and launch the interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
