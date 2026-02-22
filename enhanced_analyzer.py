
import re
import ast
import numpy as np
import time
from collections import Counter
import math
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from typing import Dict, List, Tuple, Optional
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import javalang

class EnhancedCodeAnalyzer:
    def __init__(self):
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        

        self.neural_model = None
        self.scaler = None
        self.label_encoder = None
        
       
        self.compiled_patterns = {
            'function_calls': re.compile(r'\b\w+\s*\('),
            'method_calls': re.compile(r'\.\w+'),
            'camel_case': re.compile(r'[A-Z][a-z]+'),
            'snake_case': re.compile(r'[a-z]+_[a-z]+'),
            'variables': re.compile(r'\b[a-zA-Z_]\w*\s*='),
            'add_assign': re.compile(r'\b[a-zA-Z_]\w*\s*\+='),
            'sub_assign': re.compile(r'\b[a-zA-Z_]\w*\s*-='),
            'floats': re.compile(r'\b\d+\.\d+'),
            'integers': re.compile(r'\b\d+'),
            'mixed_case': re.compile(r'[a-z][A-Z]'),
            'multiple_spaces': re.compile(r'\s{2,}'),
            'tabs': re.compile(r'\t'),
            'words': re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),
            'comments': re.compile(r'#.*$|/\*.*?\*/|//.*$', re.MULTILINE | re.DOTALL),
            'strings': re.compile(r'"[^"]*"|\'[^\']*\''),
            'imports': re.compile(r'^(import|from)\s+', re.MULTILINE),
            'classes': re.compile(r'^class\s+', re.MULTILINE),
            'functions': re.compile(r'^def\s+', re.MULTILINE),
            'decorators': re.compile(r'^@\w+', re.MULTILINE),
            'async_def': re.compile(r'^async\s+def', re.MULTILINE),
            'type_hints': re.compile(r':\s*\w+(\[\w+\])?'),
            'f_strings': re.compile(r'f["\']'),
            'list_comprehensions': re.compile(r'\[.*for.*in.*\]'),
            'lambda_functions': re.compile(r'lambda\s+'),
            'generator_expressions': re.compile(r'\(.*for.*in.*\)'),
            'context_managers': re.compile(r'with\s+'),
            'exception_handling': re.compile(r'try:|except|finally'),
            'assertions': re.compile(r'assert\s+'),
            'docstrings': re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL),
            'magic_methods': re.compile(r'__\w+__'),
            'private_methods': re.compile(r'_\w+'),
            'constants': re.compile(r'^[A-Z_][A-Z0-9_]*\s*=', re.MULTILINE)
        }
        def _temperature_scale(self, prob, T=2.5):
            """Smooths probabilities so they aren't extreme."""
            prob = np.clip(prob, 1e-8, 1 - 1e-8)
            logit = np.log(prob / (1 - prob))
            scaled = 1 / (1 + np.exp(-logit / T))
            return scaled
        

        self._load_models()
        print("Enhanced Code Analyzer initialized (Comprehensive Analysis)")

    def _detect_language(self, code: str) -> str:
        """Detect programming language using Pygments"""
        try:
            lexer = guess_lexer(code)
            return lexer.name.lower()
        except:
            return "unknown"

    def _load_models(self):

        """Load neural and preprocessing models in correct priority order."""
        try:
            model_dir = "/mnt/data"


            improved_model = os.path.join(model_dir, "improved_neural_model.pkl")
            improved_scaler = os.path.join(model_dir, "improved_scaler.pkl")
            improved_encoder = os.path.join(model_dir, "improved_label_encoder.pkl")


            simple_model = os.path.join(model_dir, "simple_neural_model.pkl")
            simple_scaler = os.path.join(model_dir, "simple_neural_scaler.pkl")
            simple_encoder = os.path.join(model_dir, "simple_neural_label_encoder.pkl")

            if os.path.exists(improved_model):
                self.neural_model = joblib.load(improved_model)
                print("Loaded: improved_neural_model.pkl")
            elif os.path.exists(simple_model):
                self.neural_model = joblib.load(simple_model)
                print("Loaded: simple_neural_model.pkl")
            else:
                print("❌ No neural model found")

        
            if os.path.exists(improved_scaler):
                self.scaler = joblib.load(improved_scaler)
                print("Loaded: improved_scaler.pkl")
            elif os.path.exists(simple_scaler):
                self.scaler = joblib.load(simple_scaler)
                print("Loaded: simple_neural_scaler.pkl")
            else:
                print("❌ No scaler found")


            if os.path.exists(improved_encoder):
                self.label_encoder = joblib.load(improved_encoder)
                print("Loaded: improved_label_encoder.pkl")
            elif os.path.exists(simple_encoder):
                self.label_encoder = joblib.load(simple_encoder)
                print("Loaded: simple_neural_label_encoder.pkl")
            else:
                print("❌ No label encoder found")

            print("✔ All models loaded successfully (with fallback support)")

        except Exception as e:
            print(f"❌ ERROR: Failed loading models → {e}")

    @lru_cache(maxsize=2000)
    def extract_features_fast(self, code: str) -> List[float]:
        start_time = time.time()
        

        basic_features = self._extract_basic_features(code)
        

        advanced_features = self._extract_advanced_features_cached(code)

        style_features = self._extract_style_features(code)
  
        enhanced_features = self._extract_enhanced_features(code)
        
       
        comprehensive_features = self._extract_comprehensive_features(code)
        
       
        features = basic_features + advanced_features + style_features + enhanced_features + comprehensive_features
     
        if len(features) < 70:
            features += [0.0] * (70 - len(features))
        elif len(features) > 70:
            features = features[:70]
        
        elapsed = time.time() - start_time
        print(f"Enhanced feature extraction completed in {elapsed:.3f}s (features: {len(features)})")
        return features

    def _extract_basic_features(self, code: str) -> List[float]:
        """Extract basic code statistics (safe + consistent)."""
        features = []


        features.append(len(code))                  # 1
        features.append(code.count("\n"))           # 2
        features.append(len(code.split()))          # 3

        alpha = sum(c.isalpha() for c in code)
        digit = sum(c.isdigit() for c in code)
        space = sum(c.isspace() for c in code)
        brackets = sum(c in "{}[]()" for c in code)
        punct = sum(c in ";:,." for c in code)
        features += [alpha, digit, space, brackets, punct]

        lines = code.split("\n")
        lengths = [len(l) for l in lines if l.strip()]
        if lengths:
            features += [                          # 9–11
                len(lines),
                float(np.mean(lengths)),
                float(np.std(lengths))
        ]
        else:
            features += [0, 0.0, 0.0]

        return features
    
    
    def _extract_java_features(self, code: str) -> List[int]:

        """Extract Java features (aligned to Python equivalent)."""
        try:
            tree = javalang.parse.parse(code)

            method_count = 0
            class_count = 0
            loop_count = 0
            cond_count = 0

            for _, node in tree:
                if isinstance(node, javalang.tree.ClassDeclaration):
                    class_count += 1
                if isinstance(node, javalang.tree.MethodDeclaration):
                    method_count += 1
                if isinstance(node, (javalang.tree.ForStatement, javalang.tree.WhileStatement)):
                    loop_count += 1
                if isinstance(node, javalang.tree.IfStatement):
                    cond_count += 1

            import_count = len(tree.imports)

       
            return [
                method_count,      # functions
                class_count,       # classes
                import_count,      # imports
                0,                 # imports_from
                0,                 # calls (not counted)
                0,                 # assignments
                loop_count,        # loops
                cond_count         # conditionals
        ]

        except:
            return [0] * 8
        
    def _extract_generic_features(self, code: str) -> List[int]:
        """Generic fallback for JS, C, C++, PHP… returns same 8 fields."""
        functions = len(re.findall(r"\w+\s*\(", code))
        classes = len(re.findall(r"class\s+\w+", code, re.IGNORECASE))
        loops = len(re.findall(r"\bfor\b|\bwhile\b", code))
        conditionals = len(re.findall(r"\bif\b", code))
        imports = len(re.findall(r"import\s+", code))

        return [
            functions,     # functions
            classes,       # classes
            imports,       # imports
            0,             # imports_from
            0,             # calls
            0,             # assignments
            loops,         # loops
            conditionals   # conditionals
    ]

    def _extract_advanced_features_cached(self, code: str) -> List[float]:
        features = []

        
        lines = code.split("\n")
        comment_lines = sum(l.strip().startswith(("#", "//", "/*")) for l in lines)
        features.append(comment_lines)                             # 1
        features.append(comment_lines / max(len(lines), 1))        # 2

   
        lang = self._detect_language(code)

        if "java" in lang:
            features += self._extract_java_features(code)
        elif "python" in lang:
            try:
                tree = ast.parse(code)
                functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                imports = len([n for n in ast.walk(tree) if isinstance(n, ast.Import)])
                imports_from = len([n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)])
                calls = len([n for n in ast.walk(tree) if isinstance(n, ast.Call)])
                assignments = len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)])
                loops = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))])
                conditionals = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
            except:
                functions = classes = imports = imports_from = calls = assignments = loops = conditionals = 0

            features += [
                functions,
                classes,
                imports,
                imports_from,
                calls,
                assignments,
                loops,
                conditionals
        ]
        else:
            features += self._extract_generic_features(code)

        return features
    
    
    def _extract_style_features(self, code: str) -> List[float]:
        features = []

        lines = code.split("\n")
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]

        if indents:
            features += [float(np.mean(indents)), float(np.std(indents)), float(max(indents))]
        else:
            features += [0.0, 0.0, 0.0]

        words = self.compiled_patterns["words"].findall(code)
        camel = sum(1 for w in words if re.match(r"^[a-z]+[A-Z]", w))
        snake = sum("_" in w for w in words)
        pascal = sum(re.match(r"^[A-Z][a-zA-Z0-9]*$", w) is not None for w in words)

        features += [camel, snake, pascal]

        features += [
            code.count("def "),
            code.count("class "),
            code.count("import "),
            code.count("from ")
    ]

        return features

    def _extract_enhanced_features(self, code: str) -> List[float]:
        """Extract enhanced features that simulate neural-like patterns (optimized)"""
        features = []
        
        
        features.append(len(self.compiled_patterns['function_calls'].findall(code)))
        features.append(len(self.compiled_patterns['method_calls'].findall(code)))
        features.append(len(self.compiled_patterns['camel_case'].findall(code)))
        features.append(len(self.compiled_patterns['snake_case'].findall(code)))
        
        patterns = ['return', 'print', 'assert', 'raise', 'with', 'async', 'await']
        for pattern in patterns:
            features.append(code.count(pattern))
        
       
        features.append(len(self.compiled_patterns['variables'].findall(code)))
        features.append(len(self.compiled_patterns['add_assign'].findall(code)))
        features.append(len(self.compiled_patterns['sub_assign'].findall(code)))
        
        
        features.append(code.count('"'))
        features.append(code.count("'"))
        features.append(code.count('f"'))
        features.append(code.count("f'"))
        
      
        features.append(len(self.compiled_patterns['floats'].findall(code)))
        features.append(len(self.compiled_patterns['integers'].findall(code)))
        
        
        control_patterns = ['break', 'continue', 'pass']
        for pattern in control_patterns:
            features.append(code.count(pattern))
        
        
        error_patterns = ['except', 'finally', 'else:']
        for pattern in error_patterns:
            features.append(code.count(pattern))
        
       
        doc_patterns = ['"""', "'''", 'TODO', 'FIXME', 'HACK']
        for pattern in doc_patterns:
            features.append(code.count(pattern))
        
       
        modern_patterns = ['lambda', 'yield', 'generator', 'decorator']
        for pattern in modern_patterns:
            features.append(code.count(pattern))
        
       
        features.append(len(self.compiled_patterns['mixed_case'].findall(code)))
        features.append(len(self.compiled_patterns['multiple_spaces'].findall(code)))
        features.append(len(self.compiled_patterns['tabs'].findall(code)))
        
        return features

    def _extract_comprehensive_features(self, code: str) -> List[float]:
        features = []

   
        features.append(len(self.compiled_patterns['type_hints'].findall(code)))
        features.append(len(self.compiled_patterns['f_strings'].findall(code)))
        features.append(len(self.compiled_patterns['list_comprehensions'].findall(code)))
        features.append(len(self.compiled_patterns['lambda_functions'].findall(code)))
        features.append(len(self.compiled_patterns['generator_expressions'].findall(code)))
        features.append(len(self.compiled_patterns['context_managers'].findall(code)))
        features.append(len(self.compiled_patterns['exception_handling'].findall(code)))
        features.append(len(self.compiled_patterns['assertions'].findall(code)))
        features.append(len(self.compiled_patterns['docstrings'].findall(code)))
        features.append(len(self.compiled_patterns['magic_methods'].findall(code)))
        features.append(len(self.compiled_patterns['private_methods'].findall(code)))
        features.append(len(self.compiled_patterns['constants'].findall(code)))

 
        char_freq = Counter(code)
        total_chars = len(code)
        if total_chars > 0:
            entropy = -sum((freq / total_chars) * math.log2(freq / total_chars)
                       for freq in char_freq.values())
            features.append(entropy)
        else:
            features.append(0)

  
        words = self.compiled_patterns['words'].findall(code)
        unique_words = len(set(words))
        features.append(unique_words / max(len(words), 1))

  
        lang = self._detect_language(code)
        if "python" in lang:
            try:
                tree = ast.parse(code)
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                if len(functions) > 0:
                    features.append(len(code) / len(functions))
                else:
                    features.append(0)
            except:
                features.append(0)
        else:
            features.append(0)

        return features
    
        

    def analyze_code_comprehensive(self, code: str) -> Dict:
        """Comprehensive code analysis with detailed insights"""
        analysis = {}
        
       
        lines = code.split('\n')
        analysis['basic_metrics'] = {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'blank_lines': len([line for line in lines if not line.strip()]),
            'total_characters': len(code),
            'average_line_length': np.mean([len(line) for line in lines if line.strip()]) if lines else 0
        }
        
        
        try:
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            analysis['complexity'] = {
                'functions': len(functions),
                'classes': len(classes),
                'imports': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
                'function_calls': len([node for node in ast.walk(tree) if isinstance(node, ast.Call)]),
                'assignments': len([node for node in ast.walk(tree) if isinstance(node, ast.Assign)]),
                'loops': len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]),
                'conditionals': len([node for node in ast.walk(tree) if isinstance(node, ast.If)]),
                'exceptions': len([node for node in ast.walk(tree) if isinstance(node, (ast.Try, ast.ExceptHandler))])
            }
        except:
            analysis['complexity'] = {
                'functions': 0, 'classes': 0, 'imports': 0, 'function_calls': 0,
                'assignments': 0, 'loops': 0, 'conditionals': 0, 'exceptions': 0
            }
        
       
        analysis['style'] = {
            'indentation_consistency': self._analyze_indentation(code),
            'naming_conventions': self._analyze_naming_conventions(code),
            'comment_ratio': analysis['basic_metrics']['comment_lines'] / max(analysis['basic_metrics']['total_lines'], 1),
            'line_length_variation': np.std([len(line) for line in lines if line.strip()]) if lines else 0
        }
        
       
        analysis['quality'] = {
            'has_docstrings': bool(self.compiled_patterns['docstrings'].findall(code)),
            'has_type_hints': bool(self.compiled_patterns['type_hints'].findall(code)),
            'has_error_handling': bool(self.compiled_patterns['exception_handling'].findall(code)),
            'has_assertions': bool(self.compiled_patterns['assertions'].findall(code)),
            'uses_modern_features': bool(self.compiled_patterns['f_strings'].findall(code) or 
                                       self.compiled_patterns['lambda_functions'].findall(code))
        }
        
        return analysis

    def _analyze_indentation(self, code: str) -> str:
        """Analyze indentation consistency"""
        lines = code.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
        
        if not indent_levels:
            return "No indentation"
        
        if len(set(indent_levels)) == 1:
            return "Consistent"
        elif len(set(indent_levels)) <= 3:
            return "Mostly consistent"
        else:
            return "Inconsistent"

    def _analyze_naming_conventions(self, code: str) -> Dict:
        """Analyze naming conventions"""
        words = self.compiled_patterns['words'].findall(code)
        
        camel_case = sum(1 for word in words if re.match(r'^[a-z]+[A-Z]', word))
        snake_case = sum(1 for word in words if '_' in word)
        pascal_case = sum(1 for word in words if re.match(r'^[A-Z][a-zA-Z0-9]*$', word))
        
        total = len(words)
        if total == 0:
            return {'camel_case': 0, 'snake_case': 0, 'pascal_case': 0, 'dominant': 'none'}
        
        conventions = {
            'camel_case': camel_case / total,
            'snake_case': snake_case / total,
            'pascal_case': pascal_case / total
        }
        
        dominant = max(conventions, key=conventions.get)
        conventions['dominant'] = dominant
        
        return conventions

    def predict(self, code: str) -> Dict:
        """Make prediction using neural model with comprehensive analysis"""
        start_time = time.time()
        features = self.extract_features_fast(code)
        
       
        comprehensive_analysis = self.analyze_code_comprehensive(code)
        
        if self.neural_model is not None and self.scaler is not None:
            try:
              
                features_scaled = self.scaler.transform([features])
                
               
                prediction = self.neural_model.predict(features_scaled)[0]
                probability = self.neural_model.predict_proba(features_scaled)[0]
                raw_ai = probability[1] if len(probability) > 1 else probability[0] 
                raw_human = probability[0] if len(probability) > 1 else (1 - raw_ai)

                scaled_ai = self._temperature_scale(raw_ai, T=2.5)
                scaled_human = 1 - scaled_ai

                confidence = max(scaled_ai, scaled_human)
                ai_probability = float(np.clip(ai_probability, 0.05, 0.95))
                human_probability = 1 - ai_probability
                confidence = max(ai_probability, human_probability)
                
                
                ai_probability = scaled_ai
                human_probability = scaled_human
                elapsed = time.time() - start_time
                
                
                
                return {
                    'prediction': 'AI' if prediction == 1 else 'Human',
                    'confidence': float(max(probability)),
                    'ai_probability': float(probability[1] if len(probability) > 1 else probability[0]),
                    'human_probability': float(probability[0] if len(probability) > 1 else 1 - probability[0]),
                    'features_used': len(features),
                    'neural_features': 80,
                    'processing_time': elapsed,
                    'comprehensive_analysis': comprehensive_analysis,
                    'model_type': 'enhanced_neural'
                }
            except Exception as e:
                print(f"Error in neural prediction: {e}")
        
        # Fallback to rule-based prediction
        return self._rule_based_prediction(code, features, comprehensive_analysis, time.time() - start_time)

    def _rule_based_prediction(self, code: str, features: List[float], analysis: Dict, elapsed: float) -> Dict:
        """Rule-based prediction as fallback with comprehensive analysis"""
       
        ai_indicators = 0
        human_indicators = 0
        
        
        if features[0] > 1000:  # Long code
            ai_indicators += 1
        if features[1] > 50:  # Many lines
            ai_indicators += 1
        if features[2] > 200:  # Many words
            ai_indicators += 1
        if analysis['style']['comment_ratio'] < 0.1:  # Few comments
            ai_indicators += 1
        if analysis['complexity']['functions'] > 5:  # Many functions
            ai_indicators += 1
        
        
        if 50 <= features[0] <= 500:  # Moderate length
            human_indicators += 1
        if features[1] < 30:  # Fewer lines
            human_indicators += 1
        if analysis['style']['comment_ratio'] > 0.2:  # More comments
            human_indicators += 1
        if analysis['style']['indentation_consistency'] == 'Consistent':
            human_indicators += 1
        if analysis['quality']['has_docstrings']:
            human_indicators += 1
        
        total_indicators = ai_indicators + human_indicators
        if total_indicators == 0:
            confidence = 0.5
            prediction = 'Human'
        else:
            confidence_raw = max(ai_indicators, human_indicators) / max(total_indicators, 1)


            confidence = 0.15 + 0.7 * confidence_raw

            if ai_indicators > human_indicators:
                prediction = "AI"
                ai_probability = confidence
                human_probability = 1 - confidence
            else:
                prediction = "Human"
                human_probability = confidence
                ai_probability = 1 - confidence
                
            ai_probability = float(np.clip(ai_probability, 0.05, 0.95))
            human_probability = 1 - ai_probability
            confidence = max(ai_probability, human_probability)

        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'ai_probability': confidence if prediction == 'AI' else 1 - confidence,
            'human_probability': confidence if prediction == 'Human' else 1 - confidence,
            'features_used': len(features),
            'neural_features': 80,
            'processing_time': elapsed,
            'comprehensive_analysis': analysis,
            'model_type': 'enhanced_rule_based'

        } 
