from flask import Flask, request, jsonify, send_file
from flask_ngrok import run_with_ngrok  # Import flask-ngrok
from pyngrok import ngrok
import threading 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import speech_recognition as sr
import tempfile
import random

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)  # This enables public access in Colab

# Define the ChatBot class
class ChatBot:
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
            "review": "Review this code for best practices, performance, and potential improvements",
            "analyze": "Analyze this code for complexity, performance implications, and potential issues",
            "optimize": "Suggest optimizations for this code",
            "refactor": "Suggest how this code could be refactored for better maintainability",
            "secure": "Analyze this code for security vulnerabilities",
            "complexity": "Analyze the time and space complexity of this code",
            "general": "Provide a general response to the query about this code"
        }

        self.supported_languages = [
            "python",
            "C++",
            "java",
            "C",
            "JavaScript"
        ]
        
        # List of random computer science topics for blog generation
        self.random_topics = [
            "Introduction to Machine Learning",
            "The Future of Quantum Computing",
            "How Blockchain Technology Works",
            "Understanding Big Data and Its Applications",
            "The Role of Artificial Intelligence in Modern Society",
            "Cybersecurity: Challenges and Solutions",
            "The Evolution of Programming Languages",
            "Cloud Computing: Benefits and Challenges",
            "The Impact of IoT on Everyday Life",
            "Ethical Considerations in AI Development"
        ]

        # Add DSA topics
        self.dsa_topics = [
            "Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Graphs",
            "Sorting", "Searching", "Dynamic Programming", "Greedy Algorithms",
            "Hash Tables", "Recursion", "Binary Search", "Heap", "Trie"
        ]

        # Define difficulty levels
        self.difficulty_levels = {
            "easy": "should be solvable within 15-20 minutes and use basic programming concepts",
            "medium": "should require 20-30 minutes and involve moderate algorithm complexity",
            "hard": "should be challenging, potentially requiring 30-45 minutes and advanced algorithmic concepts"
        }

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
        response = full_response.split("[/INST]")[-1].strip()
        return response.replace("[INST]", "").replace("[/INST]", "").strip()

    def text_to_speech(self, text):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            return fp.name

    def generate_blog(self, topic):
        prompt = self.create_prompt(f"Write a detailed blog for computer science students about {topic}.")
        return self.generate_response(prompt)

    def generate_dsa_question(self, topic, difficulty):
        """Generate a DSA question based on topic and difficulty"""
        if topic.lower() not in [t.lower() for t in self.dsa_topics]:
            return None, "Invalid topic"
        
        if difficulty.lower() not in self.difficulty_levels:
            return None, "Invalid difficulty level"

        prompt = self.create_prompt(
            f"""Create a Data Structures and Algorithms question about {topic} that {self.difficulty_levels[difficulty.lower()]}. 
            
            Format the response as follows:
            Title: [Question title]
            Difficulty: {difficulty}
            Topic: {topic}
            
            Problem Statement:
            [Detailed problem description]
            
            Input Format:
            [Description of input format]
            
            Output Format:
            [Description of expected output format]
            
            Example:
            Input: [Sample input]
            Output: [Sample output]
            
            Constraints:
            [List relevant constraints]
            
            Note:
            [Any additional hints or notes if necessary]"""
        )

        return self.generate_response(prompt), "Success"

    # New method for code execution
    def execute_code(self, code: str, language: str) -> tuple[bool, str]:
        """
        Use LLM to simulate code execution and return the expected output
        """
        if language.lower() not in self.supported_languages:
            return False, f"Language {language} is not supported"

        prompt = self.create_prompt(
            f"""You are a {language} compiler/interpreter. Given the following {language} code, 
            provide ONLY the exact output that would be produced when running this code. 
            If there are any errors, provide ONLY the error message.
            Do not provide any explanations, comments, or additional information.
            
            {language} Code:
            {code}
            
            Output:"""
        )

        try:
            output = self.generate_response(prompt)
            
            # Check if the output suggests an error
            error_indicators = [
                "error", "exception", "undefined", "syntax error",
                "cannot", "invalid", "failed", "not defined"
            ]
            
            has_error = any(indicator in output.lower() for indicator in error_indicators)
            return (not has_error, output.strip())

        except Exception as e:
            return False, str(e)

    def analyze_code(self, code: str = None, query: str = None) -> str:
        """
        Analyze code based on user query or provide programming help
        """
        if not query:
            return "Please provide a query about what you'd like to know."

        # Detect the type of analysis needed
        task_type = "general"
        for task, description in self.code_analysis_tasks.items():
            if any(keyword in query.lower() for keyword in [task, f"{task}ing", f"{task}e"]):
                task_type = task
                break

        # Create appropriate prompt based on whether code is provided
        if code:
            prompt = self.create_prompt(
                f"""As an expert programmer, help with this code-related question.

                Code:
                {code}

                Query: {query}

                Task: {self.code_analysis_tasks[task_type]}

                Please provide a detailed and specific response addressing the query and task.
                If analyzing features or issues, use line numbers when referencing specific parts of the code.
                For bugs or improvements, provide specific examples and explanations.
                Focus on practical, actionable insights."""
            )
        else:
            prompt = self.create_prompt(
                f"""As an expert programmer, help with this programming question.

                Query: {query}

                Please provide a detailed and helpful response to this programming question.
                Include specific examples where appropriate.
                Focus on practical, actionable advice and explanations."""
            )

        return self.generate_response(prompt)


# Initialize the chatbot
chatbot = ChatBot()


# New documentation HTML template
API_DOC_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Learning Assistant API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; border-bottom: 2px solid #333; }
        h2 { color: #444; margin-top: 30px; }
        .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .method { color: #fff; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
        .get { background-color: #61affe; }
        .post { background-color: #49cc90; }
        code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f5f5f5; }
    </style>
</head>
<body>
    <h1>AI Learning Assistant API Documentation</h1>
    
    <p>This API provides various AI-powered learning assistance features including code analysis, DSA problem generation, blog generation, and text-to-speech conversion.</p>

    <h2>Base URL</h2>
    <p>All endpoints are relative to: <code>[Your ngrok URL]</code></p>

    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/</code>
        <p>Returns this documentation page.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/text-to-speech</code>
        <p>Convert text to speech audio.</p>
        <h3>Request Body:</h3>
        <table>
            <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
            <tr><td>text</td><td>string</td><td>Text to convert to speech</td></tr>
        </table>
        <p>Returns: Audio file (MP3)</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/generate-random-blog</code>
        <p>Generate a blog post on a random computer science topic.</p>
        <p>Returns: JSON with topic and blog content</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/generate-blog</code>
        <h3>Request Body:</h3>
        <table>
            <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
            <tr><td>topic</td><td>string</td><td>Topic for blog generation</td></tr>
        </table>
        <p>Returns: JSON with topic and blog content</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/generate-dsa-question</code>
        <h3>Request Body:</h3>
        <table>
            <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
            <tr><td>topic</td><td>string</td><td>DSA topic (e.g., Arrays, Linked Lists)</td></tr>
            <tr><td>difficulty</td><td>string</td><td>Question difficulty (easy, medium, hard)</td></tr>
        </table>
        <p>Returns: JSON with topic, difficulty, and generated question</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/get-dsa-options</code>
        <p>Get available DSA topics and difficulty levels.</p>
        <p>Returns: JSON with lists of topics and difficulties</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/execute-code</code>
        <h3>Request Body:</h3>
        <table>
            <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
            <tr><td>code</td><td>string</td><td>Code to execute</td></tr>
            <tr><td>language</td><td>string</td><td>Programming language</td></tr>
        </table>
        <p>Returns: JSON with execution success status and output</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/supported-languages</code>
        <p>Get list of supported programming languages.</p>
        <p>Returns: JSON with list of supported languages</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span>
        <code>/analyze-code</code>
        <h3>Request Body:</h3>
        <table>
            <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
            <tr><td>code</td><td>string</td><td>(Optional) Code to analyze</td></tr>
            <tr><td>query</td><td>string</td><td>Analysis query or question</td></tr>
        </table>
        <p>Returns: JSON with analysis results</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span>
        <code>/analysis-capabilities</code>
        <p>Get available code analysis tasks.</p>
        <p>Returns: JSON with list of supported analysis tasks</p>
    </div>
</body>
</html>
"""

# Add the base route for API documentation
@app.route('/')
def api_documentation():
    return render_template_string(API_DOC_TEMPLATE)

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    audio_path = chatbot.text_to_speech(text)
    return send_file(audio_path, mimetype='audio/mp3', as_attachment=True, download_name='response.mp3')

@app.route('/generate-random-blog', methods=['GET'])
def generate_random_blog():
    topic = random.choice(chatbot.random_topics)
    return jsonify({"topic": topic, "blog": chatbot.generate_blog(topic)})

@app.route('/generate-blog', methods=['POST'])
def generate_blog():
    data = request.get_json()
    topic = data.get('topic', '')
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    return jsonify({"topic": topic, "blog": chatbot.generate_blog(topic)})

@app.route('/generate-dsa-question', methods=['POST'])
def generate_dsa_question():
    data = request.get_json()
    topic, difficulty = data.get('topic', ''), data.get('difficulty', '')

    if not topic or not difficulty:
        return jsonify({"error": "Both topic and difficulty must be provided"}), 400

    question, message = chatbot.generate_dsa_question(topic, difficulty)
    
    if question is None:
        return jsonify({"error": message}), 400

    return jsonify({"topic": topic, "difficulty": difficulty, "question": question})

@app.route('/get-dsa-options', methods=['GET'])
def get_dsa_options():
    return jsonify({"topics": chatbot.dsa_topics, "difficulties": list(chatbot.difficulty_levels.keys())})

# New endpoints for code execution
@app.route('/execute-code', methods=['POST'])
def execute_code():
    data = request.get_json()
    
    if not data or 'code' not in data or 'language' not in data:
        return jsonify({
            'success': False,
            'output': 'Both code and language must be provided'
        }), 400
    
    code = data['code']
    language = data['language'].lower()
    
    success, output = chatbot.execute_code(code, language)
    
    return jsonify({
        'success': success,
        'output': output
    })

@app.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'languages': chatbot.supported_languages
    })

@app.route('/analyze-code', methods=['POST'])
def analyze_code():
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({
            'success': False,
            'output': 'Query must be provided'
        }), 400
    
    code = data.get('code', None)  # Code is optional
    query = data['query']
    
    try:
        analysis = chatbot.analyze_code(code, query)
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'output': str(e)
        }), 500

@app.route('/analysis-capabilities', methods=['GET'])
def get_analysis_capabilities():
    return jsonify({
        'supported_tasks': chatbot.code_analysis_tasks
    })

# Run the app in Colab using flask-ngrok
if __name__ == '__main__':
    # Open an ngrok tunnel to the Flask server
    port = 8080
    public_url = ngrok.connect().public_url
    print(f"Public URL: {public_url}")

    # Start Flask app
    def run_flask():
        app.run()

    # Run Flask in a separate thread
    threading.Thread(target=run_flask).start()
