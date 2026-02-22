import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from enhanced_analyzer import EnhancedCodeAnalyzer
from werkzeug.utils import secure_filename
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import zipfile
import tempfile
import shutil
import traceback
from pygments.lexers import guess_lexer, guess_lexer_for_filename
from pygments.util import ClassNotFound


try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

app = Flask(__name__)


model_cache = {}
cache_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=8)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt', 'scala', 'html', 'css', 'xml', 'json', 'sql', 'sh', 'bat', 'ps1', 'md', 'pdf', 'zip', 'rar', '7z'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

analyzer = None

EXTENSION_LANGUAGE_MAP = {
    'py': 'Python',
    'js': 'JavaScript',
    'java': 'Java',
    'cpp': 'C++',
    'c': 'C',
    'cs': 'C#',
    'php': 'PHP',
    'rb': 'Ruby',
    'go': 'Go',
    'rs': 'Rust',
    'swift': 'Swift',
    'kt': 'Kotlin',
    'scala': 'Scala',
    'html': 'HTML',
    'css': 'CSS',
    'xml': 'XML',
    'json': 'JSON',
    'sql': 'SQL',
    'sh': 'Shell',
    'bat': 'Batch',
    'ps1': 'PowerShell',
    'md': 'Markdown'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_analyzer():
    """Initialize the enhanced analyzer"""
    global analyzer
    print("Loading enhanced analyzer...")
    analyzer = EnhancedCodeAnalyzer()
    print("Enhanced analyzer loaded successfully")

def extract_code_from_file(file_path):
    """Extract code from various file types"""
    try:
        file_ext = file_path.rsplit('.', 1)[1].lower()

        
        if file_ext == 'pdf':
            if not PyPDF2:
                return None, "PDF support not installed"

            text = ""
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)

                    for page in reader.pages:
                        try:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted + "\n"
                        except:
                            continue

                if text.strip() == "":
                    return None, "PDF has no readable text (maybe scanned)."

                return text.strip(), None

            except Exception as e:
                return None, f"PDF read error: {str(e)}"


        elif file_ext in ['zip', 'rar', '7z']:
            extracted_dir = tempfile.mkdtemp()
            try:
                if file_ext == 'zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extracted_dir)

                code_files = []
                for root, dirs, files in os.walk(extracted_dir):
                    for file in files:
                        if '.' in file:
                            ext = file.rsplit('.', 1)[1].lower()
                            if ext in ALLOWED_EXTENSIONS:
                                code_files.append(os.path.join(root, file))

                if not code_files:
                    return None, "No code files found in archive"

                combined_code = ""
                for code_file in code_files[:10]:
                    try:
                        with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                            combined_code += f"\n# File: {os.path.basename(code_file)}\n"
                            combined_code += f.read() + "\n"
                    except Exception as e:
                        combined_code += f"\n# Error reading {os.path.basename(code_file)}: {str(e)}\n"

                return combined_code.strip(), None

            finally:
                shutil.rmtree(extracted_dir, ignore_errors=True)

        # --- NORMAL FILES ---
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(), None

    except Exception as e:
        return None, f"Error reading file: {str(e)}"
def estimate_processing_time(code_length):
    """Estimate processing time based on code length"""

    base_time = 0.1
    length_factor = code_length / 1000  
    return min(base_time + length_factor, 5.0)  

def detect_language_perfect(code, filename=None):
    
    if filename and filename.lower().endswith('.py'):
        return 'Python'
 
    try:
        if filename:
            lexer = guess_lexer_for_filename(filename, code)
        else:
            lexer = guess_lexer(code)
        pygments_lang = lexer.name
      
        if not filename and pygments_lang == 'Transact-SQL':
            if any(keyword in code for keyword in ['def ', 'import ', '#', 'print(', 'self', 'class ']):
                return 'Python'
        if pygments_lang and pygments_lang != 'Text only':
            return pygments_lang
    except ClassNotFound:
        pass

    if filename and '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        return EXTENSION_LANGUAGE_MAP.get(ext, 'Unknown')
    return 'Unknown'


initialize_analyzer()

@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/health')
def health():
    model_status = "enhanced"
    if analyzer is not None and getattr(analyzer, 'neural_model', None) is not None:
        model_status += "_with_neural"
    else:
        model_status += "_fallback"
    
    return jsonify({
        "status": "ok", 
        "analyzer": model_status, 
        "features": analyzer.__dict__.get('feature_cache', {}),
        "neural_model_loaded": analyzer is not None and getattr(analyzer, 'neural_model', None) is not None,
        "enhanced_features": 80,
        "performance": "optimized",
        "file_upload": "enabled",
        "comprehensive_analysis": "enabled"
    })

@app.route('/analyze', methods=['POST'])
def analyze_code():
    try:
        start_time = time.time()
        code = None
        file_info = None
        filename = None
        
     
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected.'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename) if file.filename else 'uploaded_file'
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                
                extracted_code, error = extract_code_from_file(file_path)
                if error:
                    return jsonify({'error': error}), 400
                
                code = extracted_code
                file_info = {
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'type': filename.rsplit('.', 1)[1].lower()
                }
                

                try:
                    os.remove(file_path)
                except:
                    pass
            else:
                return jsonify({'error': 'File type not allowed.'}), 400

        else:
            data = request.get_json()
            code = data.get('code', '')
        
        if not code or not code.strip():
            return jsonify({'error': 'Please provide some code to analyze or upload a file.'}), 400

        estimated_time = estimate_processing_time(len(code))
  
        prediction_result = analyzer.predict(code) if analyzer is not None else {'prediction': 'Error', 'confidence': 0, 'ai_probability': 0, 'human_probability': 0, 'features_used': 0, 'neural_features': 0, 'processing_time': 0, 'comprehensive_analysis': {}, 'model_type': 'none'}
        
    
        detected_language = detect_language_perfect(code, filename if file_info else None)

        total_time = time.time() - start_time
        

        response = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'ai_probability': prediction_result['ai_probability'],
            'human_probability': prediction_result['human_probability'],
            'language': detected_language,
            'features_used': prediction_result['features_used'],
            'neural_features': prediction_result['neural_features'],
            'model_type': prediction_result['model_type'],
            'message': 'Enhanced analysis complete with comprehensive code insights.',
            'performance': {
                'total_time': total_time,
                'estimated_time': estimated_time,
                'feature_extraction_time': prediction_result.get('processing_time', 0),
                'prediction_time': total_time - prediction_result.get('processing_time', 0)
            },
            'comprehensive_analysis': prediction_result['comprehensive_analysis'],
            'style': prediction_result['comprehensive_analysis'].get('style', {}),
            'complexity': prediction_result['comprehensive_analysis'].get('complexity', {})
        }
        

        if file_info:
            response['file_info'] = file_info
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Internal error in /analyze: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Dedicated file upload endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) if file.filename else 'uploaded_file'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            

            try:
                extracted_code, error = extract_code_from_file(file_path)
                if error:
                    return jsonify({'error': error}), 400
            except Exception as e:
                print(f"Error extracting code from file: {e}")
                traceback.print_exc()
                return jsonify({'error': f'Error extracting code: {str(e)}'}), 500
            

            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up uploaded file: {e}")
            
        
            return analyze_code_internal(extracted_code, filename)
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

def analyze_code_internal(code, filename=None):
    """Internal function to analyze code"""
    try:
        start_time = time.time()
        

        estimated_time = estimate_processing_time(len(code))
        
      
        prediction_result = analyzer.predict(code) if analyzer is not None else {'prediction': 'Error', 'confidence': 0, 'ai_probability': 0, 'human_probability': 0, 'features_used': 0, 'neural_features': 0, 'processing_time': 0, 'comprehensive_analysis': {}, 'model_type': 'none'}
        
        
        detected_language = detect_language_perfect(code, filename)
        
       
        total_time = time.time() - start_time

        response = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'ai_probability': prediction_result['ai_probability'],
            'human_probability': prediction_result['human_probability'],
            'language': detected_language,
            'features_used': prediction_result['features_used'],
            'neural_features': prediction_result['neural_features'],
            'model_type': prediction_result['model_type'],
            'message': 'Enhanced analysis complete with comprehensive code insights.',
            'performance': {
                'total_time': total_time,
                'estimated_time': estimated_time,
                'feature_extraction_time': prediction_result.get('processing_time', 0),
                'prediction_time': total_time - prediction_result.get('processing_time', 0)
            },
            'comprehensive_analysis': prediction_result['comprehensive_analysis'],
            'style': prediction_result['comprehensive_analysis'].get('style', {}),
            'complexity': prediction_result['comprehensive_analysis'].get('complexity', {})
        }
        
     
        if filename:
            response['file_info'] = {
                'filename': filename,
                'size': len(code),
                'type': filename.rsplit('.', 1)[1].lower() if '.' in filename else 'text'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded models"""
    return jsonify({
        'neural_model_loaded': analyzer is not None and getattr(analyzer, 'neural_model', None) is not None,
        'analyzer_type': 'EnhancedCodeAnalyzer',
        'feature_extraction_methods': [
            'basic_features',
            'advanced_features',
            'style_features',
            'enhanced_features',
            'comprehensive_features'
        ],
        'total_features': 80,
        'performance': 'optimized',
        'comprehensive_analysis': 'enabled',
        'file_upload': {
            'enabled': True,
            'max_size': '50MB',
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'archive_support': ['zip', 'rar', '7z']
        }
    })

if __name__ == '__main__':
    print("Starting Enhanced AI Code Detection App...")
    print("Model Status:")
    print(f"- Neural Model: {'Loaded' if analyzer is not None and getattr(analyzer, 'neural_model', None) is not None else 'Not Available'}")
    print(f"- Enhanced Features: 80")
    print(f"- Performance: Optimized")
    print(f"- File Upload: Enabled")
    print(f"- Comprehensive Analysis: Enabled")
    print(f"- Fallback Mode: {'Active' if analyzer is not None and getattr(analyzer, 'neural_model', None) is None else 'Neural Mode'}")
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5005) 
