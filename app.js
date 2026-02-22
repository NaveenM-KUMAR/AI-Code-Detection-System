
class AICodeDetector {
    constructor() {
        console.log('AICodeDetector initializing...');
        this.initializeElements();
        this.bindEvents();
        this.uploadedFile = null;
        this.charts = {};
        console.log('AICodeDetector initialized successfully');
    }

    initializeElements() {
        // Main elements
        this.codeInput = document.getElementById('codeInput');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.fileInput = document.getElementById('fileInput');
        this.fileStatus = document.getElementById('fileStatus');
        this.codeInputGroup = document.getElementById('codeInputGroup');
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.loadingSection = document.getElementById('loadingSection');
        this.predictionResult = document.getElementById('predictionResult');
        this.codeAnalysisSection = document.getElementById('codeAnalysisSection');
        
        // Analysis elements
        this.structureMetrics = document.getElementById('structureMetrics');
        this.qualityMetrics = document.getElementById('qualityMetrics');
        this.complexityMetrics = document.getElementById('complexityMetrics');
        this.styleMetrics = document.getElementById('styleMetrics');
        this.codeInsights = document.getElementById('codeInsights');
        this.codePreview = document.getElementById('codePreview');
        
        console.log('Elements found:', {
            codeInput: !!this.codeInput,
            analyzeBtn: !!this.analyzeBtn,
            resultsSection: !!this.resultsSection,
            loadingSection: !!this.loadingSection,
            predictionResult: !!this.predictionResult,
            codeAnalysisSection: !!this.codeAnalysisSection
        });
    }

    bindEvents() {
        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeCode());
        
        // Enter key in textarea
        this.codeInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.analyzeCode();
            }
        });
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }
    }

    handleFileUpload(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadedFile = file;
            this.codeInputGroup.style.display = 'none';
            
            // Get file extension
            const fileName = file.name;
            const fileExtension = fileName.split('.').pop().toLowerCase();
            
            // Show appropriate status message
            if (fileExtension === 'pdf') {
                this.fileStatus.textContent = `PDF file "${fileName}" loaded. Click Analyze Code to extract and analyze.`;
            } else {
                this.fileStatus.textContent = `${fileExtension.toUpperCase()} file "${fileName}" loaded. Click Analyze Code to analyze.`;
            }
            this.fileStatus.style.display = 'block';
        } else {
            this.uploadedFile = null;
            this.codeInputGroup.style.display = 'block';
            this.fileStatus.textContent = '';
            this.fileStatus.style.display = 'none';
        }
    }

    async analyzeCode() {
        this.showLoading(true);
        this.hideResults();
        let response, result;
        
        if (this.uploadedFile) {
            const formData = new FormData();
            formData.append('file', this.uploadedFile);
            response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
        } else {
            const code = this.codeInput.value.trim();
            
            if (!code) {
                this.showAlert('Please enter some code to analyze or upload a file.', 'danger');
                this.showLoading(false);
                return;
            }

            response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code: code })
            });
        }

        try {
            result = await response.json();
            console.log('API Response:', result);

            if (response.ok) {
                this.displayResults(result);
            } else {
                this.showAlert(result.error || 'Analysis failed.', 'danger');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showAlert('Network error. Please try again.', 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        // Show results section
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('fade-in');

        // Show file information if available
        if (result.file_info) {
            this.displayFileInfo(result.file_info);
        }

        // Update prediction
        const predictionText = this.predictionResult.querySelector('.prediction-text');
        predictionText.textContent = result.prediction;
        predictionText.className = `prediction-text`;

        // Update confidence bar
        const confidenceFill = this.predictionResult.querySelector('.confidence-fill');
        const confidenceText = this.predictionResult.querySelector('.confidence-text');
        const confidencePercent = Math.round(result.confidence * 100);
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceText.textContent = `${confidencePercent}% confidence`;

        // Show AI/Human-likeness visually
        this.displayAILikenessCircle(confidencePercent, result.prediction);

        // Show class probabilities
        this.displayClassProbabilities(result.class_probs, result.prediction);
        
        // Show prediction explanation
        this.displayPredictionExplanation(result);
        
        // Show comprehensive code analysis
        this.displayCodeAnalysis(result.features, result);
        
        // Show code preview
        this.displayCodePreview();
    }

    displayFileInfo(fileInfo) {
        let container = document.getElementById('fileInfo');
        if (!container) {
            container = document.createElement('div');
            container.id = 'fileInfo';
            container.className = 'mb-3';
            this.resultsSection.insertBefore(container, this.resultsSection.firstChild);
        }
        
        const fileIcon = fileInfo.type === 'PDF' ? 'fas fa-file-pdf' : 'fas fa-file-code';
        const fileSize = this.formatFileSize(fileInfo.size);
        
        let extractionInfo = '';
        if (fileInfo.extraction_info) {
            extractionInfo = `
                <div class="mt-2">
                    <small class="text-success">
                        <i class="fas fa-check-circle me-1"></i>
                        ${fileInfo.extraction_info}
                    </small>
                </div>
            `;
        }
        
        container.innerHTML = `
            <div class="alert alert-info">
                <h6><i class="${fileIcon} me-2"></i>File Information</h6>
                <div class="row">
                    <div class="col-md-6">
                        <strong>File Name:</strong> ${fileInfo.name}<br>
                        <strong>File Type:</strong> ${fileInfo.type}
                    </div>
                    <div class="col-md-6">
                        <strong>Content Size:</strong> ${fileSize}<br>
                        <strong>Language Detected:</strong> ${fileInfo.language || 'Unknown'}
                    </div>
                </div>
                ${extractionInfo}
                <div class="mt-2">
                    <small class="text-muted">
                        <i class="fas fa-info-circle me-1"></i>
                        ${fileInfo.type === 'PDF' ? 'Enhanced code extraction applied to PDF content.' : 'Code content was preprocessed and analyzed for optimal detection.'}
                    </small>
                </div>
            </div>
        `;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    displayAILikenessCircle(aiPercent, prediction) {
        let container = document.getElementById('aiLikenessCircle');
        if (!container) {
            container = document.createElement('div');
            container.id = 'aiLikenessCircle';
            container.className = 'my-3 d-flex justify-content-center';
            this.predictionResult.appendChild(container);
        }
        container.innerHTML = '';
        const isHuman = prediction === 'Human';
        const color = isHuman ? '#27ae60' : '#e74c3c';
        const icon = isHuman ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        const label = isHuman ? 'Human' : 'AI';
        container.innerHTML = `
            <div class="text-center">
                <svg width="120" height="120" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" fill="#f4f4f4" stroke="#ddd" stroke-width="4" />
                    <circle id="aiLikenessArc" cx="60" cy="60" r="40" fill="none" stroke="${color}" stroke-width="8" stroke-dasharray="251.2" stroke-dashoffset="0" transform="rotate(-90 60 60)" />
                    <text id="aiLikenessText" x="60" y="65" text-anchor="middle" font-size="28" font-weight="bold" fill="${color}">${aiPercent}%</text>
                </svg>
                <div class="mt-2 fw-bold" style="font-size: 1.2rem; color: ${color};">${icon} ${label} Likelihood</div>
            </div>
        `;
        // Animate the arc
        const arc = container.querySelector('#aiLikenessArc');
        const circumference = 2 * Math.PI * 40;
        const offset = circumference * (1 - aiPercent / 100);
        arc.setAttribute('stroke-dasharray', circumference);
        arc.setAttribute('stroke-dashoffset', offset);
        arc.setAttribute('stroke', color);
    }

    displayPredictionExplanation(result) {
        console.log('Displaying prediction explanation for:', result);
        
        const explanationDiv = document.getElementById('predictionExplanation');
        if (!explanationDiv) {
            console.error('Prediction explanation div not found');
            return;
        }
        
        let explanation = '';
        const confidence = (result.confidence * 100).toFixed(1);
        
        if (result.prediction === 'Human') {
            explanation = `
                <div class="alert alert-success">
                    <h6><i class="fas fa-user me-2"></i>Human-Written Code Detected</h6>
                    <p class="mb-2">This code appears to be written by a human developer.</p>
                    <ul class="mb-2">
                        <li>Natural coding patterns and personal style detected</li>
                        <li>Variable naming may be less standardized</li>
                        <li>Comments and documentation style is more informal</li>
                        <li>Code structure shows individual developer preferences</li>
                    </ul>
                    <strong>Confidence: ${confidence}%</strong>
                </div>
            `;
        } else {
            // For any AI model
            explanation = `
                <div class="alert alert-warning">
                    <h6><i class="fas fa-robot me-2"></i>AI-Generated Code Detected</h6>
                    <p class="mb-2">This code appears to be generated by an AI model (${result.prediction}).</p>
                    <ul class="mb-2">
                        <li>Consistent formatting and structure patterns</li>
                        <li>Standard naming conventions and best practices</li>
                        <li>Comprehensive documentation and comments</li>
                        <li>Clean, optimized code structure</li>
                    </ul>
                    <strong>Confidence: ${confidence}%</strong>
                </div>
            `;
        }
        
        explanationDiv.innerHTML = explanation;
        console.log('Prediction explanation displayed');
    }

    displayCodeAnalysis(features, result) {
        console.log('Displaying comprehensive code analysis for features:', features);
        
        // Show the analysis section
        this.codeAnalysisSection.style.display = 'block';
        
        if (!features || features.length === 0) {
            this.showAlert('Detailed code analysis not available for this snippet.', 'info');
            return;
        }
        
        // Display structure metrics
        this.displayStructureMetrics(features);
        
        // Display quality metrics
        this.displayQualityMetrics(features);
        
        // Display complexity metrics
        this.displayComplexityMetrics(features);
        
        // Display style metrics
        this.displayStyleMetrics(features);
        
        // Display code insights
        this.displayCodeInsights(features, result);
        
        // Display code preview
        this.displayCodePreview();
        
        // Create charts with delay to ensure DOM elements are ready
        setTimeout(() => {
            this.createCharts(features, result);
        }, 100);
    }

    displayStructureMetrics(features) {
        const metrics = [
            { label: 'Total Lines', value: features[5] || 0, unit: 'lines' },
            { label: 'Code Lines', value: features[5] - (features[6] || 0) - (features[7] || 0), unit: 'lines' },
            { label: 'Comment Lines', value: features[7] || 0, unit: 'lines' },
            { label: 'Empty Lines', value: features[6] || 0, unit: 'lines' },
            { label: 'Functions', value: features[18] || 0, unit: 'functions' },
            { label: 'Classes', value: features[19] || 0, unit: 'classes' },
            { label: 'Imports', value: features[20] || 0, unit: 'imports' },
            { label: 'Variables', value: features[21] || 0, unit: 'variables' }
        ];
        
        this.structureMetrics.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <span class="metric-label">${metric.label}</span>
                <span class="metric-value">${metric.value} ${metric.unit}</span>
            </div>
        `).join('');
    }

    displayQualityMetrics(features) {
        const commentRatio = features[5] > 0 ? ((features[7] || 0) / features[5] * 100).toFixed(1) : 0;
        const avgLineLength = features[1] || 0;
        const maxLineLength = features[2] || 0;
        
        const metrics = [
            { label: 'Comment Ratio', value: `${commentRatio}%`, quality: this.getQualityLevel(commentRatio, 10, 30) },
            { label: 'Average Line Length', value: avgLineLength.toFixed(1), quality: this.getQualityLevel(avgLineLength, 40, 80) },
            { label: 'Max Line Length', value: maxLineLength, quality: this.getQualityLevel(maxLineLength, 80, 120) },
            { label: 'Indentation Consistency', value: `${(features[38] || 0).toFixed(1)}%`, quality: this.getQualityLevel(features[38] || 0, 70, 90) },
            { label: 'Naming Convention Score', value: `${(features[39] || 0).toFixed(1)}%`, quality: this.getQualityLevel(features[39] || 0, 70, 90) }
        ];
        
        this.qualityMetrics.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <span class="metric-label">${metric.label}</span>
                <span class="metric-value ${metric.quality}">${metric.value}</span>
            </div>
        `).join('');
    }

    displayComplexityMetrics(features) {
        const complexityScore = features[35] || 0;
        const nestingDepth = features[36] || 0;
        const functionCount = features[18] || 0;
        const loopCount = features[22] || 0;
        const conditionalCount = features[23] || 0;
        const tryExceptCount = features[24] || 0;
        const avgFunctionLength = features[37] || 0;
        const maxFunctionLength = features[38] || 0;
        
        const cyclomaticComplexity = functionCount + loopCount + conditionalCount;
        const functionComplexity = functionCount > 0 ? complexityScore / functionCount : 0;
        
        const metrics = [
            { label: 'Overall Complexity Score', value: complexityScore.toFixed(1), complexity: this.getComplexityLevel(complexityScore, 5, 15), description: 'Combined complexity measure' },
            { label: 'Maximum Nesting Depth', value: nestingDepth.toString(), complexity: this.getComplexityLevel(nestingDepth, 3, 6), description: 'Deepest code nesting level' },
            { label: 'Cyclomatic Complexity', value: cyclomaticComplexity.toString(), complexity: this.getComplexityLevel(cyclomaticComplexity, 5, 15), description: 'Number of decision points' },
            { label: 'Average Function Complexity', value: functionComplexity.toFixed(1), complexity: this.getComplexityLevel(functionComplexity, 2, 5), description: 'Complexity per function' },
            { label: 'Control Structures', value: `${loopCount} loops, ${conditionalCount} conditions`, complexity: 'medium', description: 'Loop and conditional counts' },
            { label: 'Error Handling', value: tryExceptCount.toString(), complexity: this.getComplexityLevel(tryExceptCount, 1, 3), description: 'Try-except blocks' },
            { label: 'Function Size', value: `${avgFunctionLength.toFixed(1)} avg, ${maxFunctionLength} max lines`, complexity: this.getComplexityLevel(avgFunctionLength, 10, 30), description: 'Function length metrics' }
        ];
        
        this.complexityMetrics.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <span class="metric-label">${metric.label}</span>
                        <small class="text-muted d-block">${metric.description}</small>
                    </div>
                    <span class="metric-value ${metric.complexity}">${metric.value}</span>
                </div>
            </div>
        `).join('');
    }

    displayStyleMetrics(features) {
        const vocabularyRichness = features[37] || 0;
        const tokenEntropy = features[40] || 0;
        const punctuationRatio = features[41] || 0;
        const indentationConsistency = features[38] || 0;
        const namingConventionScore = features[39] || 0;
        const codeRepetitionScore = features[42] || 0;
        const structureRegularity = features[43] || 0;
        
        const codeDensity = features[5] > 0 ? ((features[5] - (features[6] || 0) - (features[7] || 0)) / features[5] * 100) : 0;
        
        const metrics = [
            { label: 'Vocabulary Richness', value: vocabularyRichness.toFixed(2), quality: this.getQualityLevel(vocabularyRichness, 0.3, 0.7), description: 'Diversity of identifiers and keywords' },
            { label: 'Token Entropy', value: tokenEntropy.toFixed(2), quality: this.getQualityLevel(tokenEntropy, 2.0, 4.0), description: 'Information content of code tokens' },
            { label: 'Punctuation Ratio', value: `${(punctuationRatio * 100).toFixed(1)}%`, quality: this.getQualityLevel(punctuationRatio * 100, 5, 15), description: 'Use of punctuation symbols' },
            { label: 'Code Density', value: `${codeDensity.toFixed(1)}%`, quality: this.getQualityLevel(codeDensity, 60, 85), description: 'Percentage of non-empty, non-comment lines' },
            { label: 'Indentation Consistency', value: `${indentationConsistency.toFixed(1)}%`, quality: this.getQualityLevel(indentationConsistency, 70, 90), description: 'Consistency of code indentation' },
            { label: 'Naming Convention Score', value: `${namingConventionScore.toFixed(1)}%`, quality: this.getQualityLevel(namingConventionScore, 70, 90), description: 'Adherence to naming conventions' },
            { label: 'Code Repetition Score', value: `${codeRepetitionScore.toFixed(1)}%`, quality: this.getQualityLevel(codeRepetitionScore, 10, 30), description: 'Level of code duplication' },
            { label: 'Structure Regularity', value: `${structureRegularity.toFixed(1)}%`, quality: this.getQualityLevel(structureRegularity, 70, 90), description: 'Consistency of code structure' }
        ];
        
        this.styleMetrics.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <span class="metric-label">${metric.label}</span>
                        <small class="text-muted d-block">${metric.description}</small>
                    </div>
                    <span class="metric-value ${metric.quality}">${metric.value}</span>
                </div>
            </div>
        `).join('');
    }

    displayCodeInsights(features, result) {
        const insights = [];
        
        // Code size insights
        const totalLines = features[5] || 0;
        const codeLines = features[5] - (features[6] || 0) - (features[7] || 0);
        if (totalLines < 10) {
            insights.push({ icon: 'fas fa-info-circle', text: 'Small code snippet - may be harder to analyze accurately', type: 'info' });
        } else if (totalLines > 100) {
            insights.push({ icon: 'fas fa-check-circle', text: 'Large codebase - analysis should be more reliable', type: 'success' });
        }
        
        // Code structure insights
        const functionCount = features[18] || 0;
        const classCount = features[19] || 0;
        if (functionCount > 5) {
            insights.push({ icon: 'fas fa-cogs', text: `Well-structured code with ${functionCount} functions`, type: 'success' });
        }
        if (classCount > 0) {
            insights.push({ icon: 'fas fa-cube', text: `Object-oriented design with ${classCount} class(es)`, type: 'info' });
        }
        
        // Complexity insights
        const complexityScore = features[35] || 0;
        if (complexityScore > 15) {
            insights.push({ icon: 'fas fa-exclamation-triangle', text: 'High complexity detected - may indicate sophisticated logic', type: 'warning' });
        } else if (complexityScore < 3) {
            insights.push({ icon: 'fas fa-thumbs-up', text: 'Low complexity - code is easy to understand and maintain', type: 'success' });
        }
        
        // Style insights
        const commentRatio = features[5] > 0 ? (features[7] || 0) / features[5] : 0;
        if (commentRatio > 0.3) {
            insights.push({ icon: 'fas fa-comments', text: 'Well-documented code with good comment coverage', type: 'success' });
        } else if (commentRatio < 0.05) {
            insights.push({ icon: 'fas fa-exclamation-circle', text: 'Low comment coverage - may affect maintainability', type: 'warning' });
        }
        
        // Naming and style insights
        const namingScore = features[39] || 0;
        if (namingScore > 80) {
            insights.push({ icon: 'fas fa-tag', text: 'Excellent naming conventions and code style', type: 'success' });
        }
        
        const indentationScore = features[38] || 0;
        if (indentationScore > 90) {
            insights.push({ icon: 'fas fa-align-left', text: 'Consistent and clean indentation throughout', type: 'success' });
        }
        
        // AI vs Human patterns
        if (result.prediction === 'Human') {
            insights.push({ icon: 'fas fa-user', text: 'Code shows natural human coding patterns and style variations', type: 'info' });
            if (result.confidence > 0.8) {
                insights.push({ icon: 'fas fa-user-check', text: 'High confidence in human-written code detection', type: 'success' });
            }
        } else {
            insights.push({ icon: 'fas fa-robot', text: 'Code exhibits consistent AI-generated patterns and formatting', type: 'info' });
            if (result.confidence > 0.8) {
                insights.push({ icon: 'fas fa-robot', text: `High confidence in AI detection (${result.prediction})`, type: 'warning' });
            }
        }
        
        // Language-specific insights
        const language = features[0];
        if (language && language !== 'Other/Unsupported') {
            insights.push({ icon: 'fas fa-code', text: `Detected programming language: ${language}`, type: 'info' });
            
            // Language-specific patterns
            if (language === 'Python') {
                if (features[30] > 0) { // type hints
                    insights.push({ icon: 'fas fa-tags', text: 'Uses type hints - modern Python practices', type: 'success' });
                }
                if (features[31] > 0) { // docstrings
                    insights.push({ icon: 'fas fa-file-alt', text: 'Includes docstrings - good documentation practices', type: 'success' });
                }
            } else if (language === 'JavaScript') {
                if (features[29] > 0) { // async functions
                    insights.push({ icon: 'fas fa-bolt', text: 'Uses async/await - modern JavaScript patterns', type: 'success' });
                }
            }
        }
        
        // File upload specific insights
        if (this.uploadedFile) {
            const fileExtension = this.uploadedFile.name.split('.').pop().toLowerCase();
            insights.push({ icon: 'fas fa-file-upload', text: `Analyzed from ${fileExtension.toUpperCase()} file`, type: 'info' });
        }
        
        this.codeInsights.innerHTML = insights.map(insight => `
            <div class="insight-item">
                <i class="${insight.icon} insight-icon text-${insight.type}"></i>
                <span>${insight.text}</span>
            </div>
        `).join('');
    }

    displayCodePreview() {
        let code = '';
        
        // If we have an uploaded file, show file information
        if (this.uploadedFile) {
            const fileName = this.uploadedFile.name;
            const fileExtension = fileName.split('.').pop().toLowerCase();
            
            if (fileExtension === 'pdf') {
                code = `PDF file "${fileName}" was analyzed.\n\nCode extracted from PDF content:\n[Content length: ${this.uploadedFile.size} bytes]\n\nNote: PDF content has been extracted and analyzed for AI/Human detection.`;
            } else {
                code = `${fileExtension.toUpperCase()} file "${fileName}" was analyzed.\n\nFile content:\n[Content length: ${this.uploadedFile.size} bytes]\n\nNote: Code content has been extracted and analyzed for AI/Human detection.`;
            }
        } else {
            code = this.codeInput.value.trim();
        }
        
        if (code) {
            // Truncate if too long
            const previewCode = code.length > 500 ? code.substring(0, 500) + '...' : code;
            this.codePreview.innerHTML = `<pre>${this.escapeHtml(previewCode)}</pre>`;
        } else {
            this.codePreview.innerHTML = '<em>No code to preview</em>';
        }
    }

    createCharts(features, result) {
        // Complexity distribution chart
        this.createComplexityChart(features);
        
        // Feature comparison chart
        this.createFeatureChart(features);
        
        // Model confidence chart
        this.createConfidenceChart(result);
    }

    createComplexityChart(features) {
        const ctx = document.getElementById('complexityChart');
        if (!ctx) return;
        
        if (this.charts.complexity) {
            this.charts.complexity.destroy();
        }
        
        const complexityScore = features[35] || 0;
        const nestingDepth = features[36] || 0;
        const functionCount = features[18] || 0;
        const loopCount = features[22] || 0;
        const conditionalCount = features[23] || 0;
        
        // Determine complexity level
        let complexityLevel = 'Low';
        let color = '#28a745';
        if (complexityScore >= 15) {
            complexityLevel = 'High';
            color = '#dc3545';
        } else if (complexityScore >= 5) {
            complexityLevel = 'Medium';
            color = '#ffc107';
        }
        
        this.charts.complexity = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Complexity Score', 'Nesting Depth', 'Control Structures'],
                datasets: [{
                    data: [complexityScore, nestingDepth, loopCount + conditionalCount],
                    backgroundColor: [color, '#17a2b8', '#6f42c1'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { 
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                return `${label}: ${value}`;
                            }
                        }
                    }
                }
            }
        });
    }

    createFeatureChart(features) {
        const ctx = document.getElementById('featureChart');
        if (!ctx) return;
        
        if (this.charts.feature) {
            this.charts.feature.destroy();
        }
        
        const featureNames = [
            'Line Length', 
            'Comments', 
            'Functions', 
            'Complexity', 
            'Style',
            'Structure',
            'Documentation'
        ];
        
        // Normalize values for better visualization
        const avgLineLength = Math.min((features[1] || 0) / 2, 10); // Normalize to 0-10
        const commentRatio = Math.min((features[7] || 0) / (features[5] || 1) * 20, 10); // Normalize to 0-10
        const functionCount = Math.min((features[18] || 0) / 2, 10); // Normalize to 0-10
        const complexityScore = Math.min((features[35] || 0) / 3, 10); // Normalize to 0-10
        const vocabularyRichness = (features[37] || 0) * 10; // Scale to 0-10
        const structureScore = Math.min(((features[18] || 0) + (features[19] || 0)) / 3, 10); // Functions + Classes
        const documentationScore = Math.min(((features[31] || 0) + (features[32] || 0) + (features[33] || 0)) * 2, 10); // Docstrings + comments
        
        const featureValues = [
            avgLineLength,
            commentRatio,
            functionCount,
            complexityScore,
            vocabularyRichness,
            structureScore,
            documentationScore
        ];
        
        this.charts.feature = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: featureNames,
                datasets: [{
                    label: 'Code Quality Metrics',
                    data: featureValues,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: { 
                        beginAtZero: true,
                        max: 10,
                        ticks: {
                            stepSize: 2
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                return `${label}: ${value.toFixed(1)}/10`;
                            }
                        }
                    }
                }
            }
        });
    }

    createConfidenceChart(result) {
        const ctx = document.getElementById('confidenceChart');
        if (!ctx) return;
        
        if (this.charts.confidence) {
            this.charts.confidence.destroy();
        }
        
        const labels = Object.keys(result.class_probs || {});
        const values = Object.values(result.class_probs || {});
        
        this.charts.confidence = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence',
                    data: values,
                    backgroundColor: labels.map(label => 
                        label === result.prediction ? 'rgba(220, 53, 69, 0.8)' : 'rgba(102, 126, 234, 0.6)'
                    ),
                    borderColor: labels.map(label => 
                        label === result.prediction ? 'rgba(220, 53, 69, 1)' : 'rgba(102, 126, 234, 1)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1 }
                }
            }
        });
    }

    getQualityLevel(value, low, high) {
        if (value < low) return 'complexity-low';
        if (value > high) return 'complexity-high';
        return 'complexity-medium';
    }

    getComplexityLevel(value, low, high) {
        if (value < low) return 'complexity-low';
        if (value > high) return 'complexity-high';
        return 'complexity-medium';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    displayClassProbabilities(classProbs, topClass) {
        // This method is already implemented in the original file
        // Keeping it for compatibility
    }

    showLoading(show) {
        this.loadingSection.style.display = show ? 'block' : 'none';
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
        this.codeAnalysisSection.style.display = 'none';
    }

    showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top of the main content
        const mainContent = document.querySelector('.main-content');
        mainContent.insertBefore(alertDiv, mainContent.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AICodeDetector();
}); 
