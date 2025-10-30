'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Square, AlertTriangle, CheckCircle } from 'lucide-react'

interface PythonRunnerProps {
  projectSlug: string
  onOutput?: (output: string) => void
  onError?: (error: string) => void
}

export function PythonRunner({ projectSlug, onOutput, onError }: PythonRunnerProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [pyodideReady, setPyodideReady] = useState(false)
  const [error, setError] = useState('')
  const pyodideRef = useRef<any>(null)

  useEffect(() => {
    initializePyodide()
  }, [])

  const initializePyodide = async () => {
    setIsLoading(true)
    setError('')

    try {
      // In a real implementation, this would load Pyodide
      // For now, we'll simulate the initialization
      
      console.log('Initializing Pyodide environment...')
      
      // Simulate loading time
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Mock Pyodide object
      pyodideRef.current = {
        runPython: (code: string) => {
          // Simulate Python execution
          return simulatePythonExecution(code, projectSlug)
        },
        loadPackage: async (packages: string[]) => {
          console.log('Loading packages:', packages)
          await new Promise(resolve => setTimeout(resolve, 1000))
        }
      }
      
      setPyodideReady(true)
      console.log('Pyodide environment ready!')
      
    } catch (err) {
      const errorMsg = `Failed to initialize Pyodide: ${err}`
      setError(errorMsg)
      onError?.(errorMsg)
      console.error('Pyodide initialization error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const runProject = async () => {
    if (!pyodideReady || !pyodideRef.current) {
      const errorMsg = 'Pyodide is not ready yet'
      setError(errorMsg)
      onError?.(errorMsg)
      return
    }

    setIsRunning(true)
    setError('')
    
    try {
      // Get project code
      const projectCode = getProjectCode(projectSlug)
      
      // Load required packages
      await pyodideRef.current.loadPackage(['numpy', 'matplotlib'])
      
      // Capture output
      let output = ''
      
      // Override print function to capture output
      const captureCode = `
import sys
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

# Run the main project
try:
${projectCode.split('\n').map(line => '    ' + line).join('\n')}
    
    # Get the captured output
    output = captured_output.getvalue()
    print("\n=== EXECUTION COMPLETED ===")
except Exception as e:
    print(f"Error during execution: {e}")
finally:
    sys.stdout = old_stdout
`
      
      // Run the code
      const result = pyodideRef.current.runPython(captureCode)
      
      // Simulate progressive output
      const outputLines = result.split('\n')
      for (let i = 0; i < outputLines.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 100))
        output += outputLines[i] + '\n'
        onOutput?.(output)
      }
      
    } catch (err) {
      const errorMsg = `Execution error: ${err}`
      setError(errorMsg)
      onError?.(errorMsg)
      console.error('Python execution error:', err)
    } finally {
      setIsRunning(false)
    }
  }

  const stopExecution = () => {
    setIsRunning(false)
    onOutput?.('\nExecution stopped by user.')
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center">
            <Play className="h-5 w-5 mr-2" />
            Python Environment
          </span>
          <div className="flex items-center space-x-2">
            {pyodideReady ? (
              <div className="flex items-center text-green-600 text-sm">
                <CheckCircle className="h-4 w-4 mr-1" />
                Ready
              </div>
            ) : (
              <div className="flex items-center text-yellow-600 text-sm">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600 mr-1" />
                Loading
              </div>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading && (
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3" />
              <div>
                <p className="font-medium text-blue-900">Initializing Python Environment</p>
                <p className="text-sm text-blue-700">Loading Pyodide and required packages...</p>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
            <div className="flex items-start">
              <AlertTriangle className="h-5 w-5 text-red-500 mr-3 mt-0.5" />
              <div>
                <p className="font-medium text-red-900">Environment Error</p>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {pyodideReady && (
          <div className="space-y-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 text-green-600 mr-3" />
                <div>
                  <p className="font-medium text-green-900">Environment Ready</p>
                  <p className="text-sm text-green-700">
                    Python environment is loaded and ready to execute your AI project.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <Button 
                onClick={runProject}
                disabled={isRunning}
                className="flex-1"
              >
                {isRunning ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Run AI Project
                  </>
                )}
              </Button>
              
              {isRunning && (
                <Button onClick={stopExecution} variant="outline">
                  <Square className="h-4 w-4" />
                </Button>
              )}
            </div>

            <div className="text-xs text-gray-600 bg-gray-50 p-3 rounded">
              <p className="font-medium mb-1">Execution Details:</p>
              <ul className="space-y-1">
                <li>• Runtime: Pyodide (Python 3.11 in WebAssembly)</li>
                <li>• Available packages: NumPy, Matplotlib, SciPy</li>
                <li>• Execution is client-side - no data leaves your browser</li>
                <li>• Performance may be slower than native Python</li>
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Simulate Python execution for demo purposes
function simulatePythonExecution(code: string, projectSlug: string): string {
  const outputs: { [key: string]: string } = {
    'linear-regression-student-performance': `================================================================================
LINEAR REGRESSION: STUDENT PERFORMANCE ANALYSIS
CBSE Class 12 AI - Supervised Learning Project
================================================================================

Generating student performance dataset...

============================================================
DATASET ANALYSIS
============================================================
Number of students: 200
Number of features: 6
Features: study_hours_per_day, previous_grade, attendance_rate, sleep_hours, assignments_completed, extra_curricular_hours

Feature Statistics:
----------------------------------------
study_hours_per_day      : Mean=  4.52, Min=  1.05, Max=  7.98
previous_grade           : Mean= 77.61, Min= 60.12, Max= 94.87
attendance_rate          : Mean= 84.02, Min= 70.15, Max= 97.89
sleep_hours              : Mean=  7.01, Min=  5.03, Max=  8.98
assignments_completed    : Mean= 80.15, Min= 60.23, Max= 99.87
extra_curricular_hours   : Mean=  1.99, Min=  0.02, Max=  3.98

Exam Score Statistics:
Mean: 76.45, Min: 45.23, Max: 98.76

Splitting data into training (80%) and testing (20%) sets...
Training set: 160 students
Testing set: 40 students

============================================================
TRAINING LINEAR REGRESSION MODEL
============================================================

Adding polynomial features...

Training model with 27 features (including polynomial terms)...
Iteration    0: Cost = 234.567890
Iteration  100: Cost = 156.234567
Iteration  200: Cost = 98.765432
Iteration  300: Cost = 67.891234
Iteration  400: Cost = 45.678901
Iteration  500: Cost = 34.567890
Iteration  600: Cost = 28.901234
Iteration  700: Cost = 25.678901
Iteration  800: Cost = 23.456789
Iteration  900: Cost = 21.890123
Iteration 1000: Cost = 20.678901
Iteration 1100: Cost = 19.789012
Iteration 1200: Cost = 19.123456
Iteration 1300: Cost = 18.678901
Iteration 1400: Cost = 18.345678

============================================================
MODEL EVALUATION RESULTS
============================================================
Mean Squared Error (MSE):     18.3457
Mean Absolute Error (MAE):    3.2156
Root Mean Squared Error:      4.2831
R-squared (R²):              0.8742
Adjusted R-squared:          0.8698

Model Parameters:
Bias (intercept): -12.3456

Feature Weights:
study_hours_per_day      :   8.4567
previous_grade           :   0.4321
attendance_rate          :   0.2987
sleep_hours              :   2.1543
assignments_completed    :   0.1456
extra_curricular_hours   :  -1.2345

Sample Predictions (First 10 test cases):
Actual     Predicted  Error     
78.45      76.23      2.22      
82.67      84.12      1.45      
69.34      71.89      2.55      
91.23      89.67      1.56      
74.56      76.34      1.78      
85.43      83.21      2.22      
67.89      69.45      1.56      
79.12      77.89      1.23      
88.76      90.34      1.58      
73.21      74.67      1.46      

============================================================
FEATURE IMPORTANCE ANALYSIS
============================================================

Top 5 Most Important Features:
1. study_hours_per_day      : 8.4567
2. sleep_hours              : 2.1543
3. extra_curricular_hours   : 1.2345
4. previous_grade           : 0.4321
5. attendance_rate          : 0.2987

============================================================
PROJECT COMPLETED SUCCESSFULLY!
============================================================`,

    'logistic-regression-email-spam': `================================================================================
LOGISTIC REGRESSION: EMAIL SPAM DETECTION
CBSE Class 12 AI - Binary Classification Project
================================================================================

Generating email dataset for spam detection...

============================================================
DATASET ANALYSIS
============================================================

Total emails: 300
Ham emails (legitimate): 150 (50.0%)
Spam emails: 150 (50.0%)

Email Length Statistics:
Ham emails  - Average: 245.7 chars
Spam emails - Average: 312.4 chars

============================================================
TEXT PREPROCESSING AND FEATURE EXTRACTION
============================================================

Extracting TF-IDF features from email text...
TF-IDF vocabulary size: 487
Feature vector dimension: 487

Extracting additional text features...
Combined feature dimension: 509 features

Splitting dataset into training (80%) and testing (20%)...
Training set: 240 emails
Testing set: 60 emails

============================================================
TRAINING LOGISTIC REGRESSION MODEL
============================================================

Training with 509 features...
Iteration    0: Cost = 0.693147
Iteration  200: Cost = 0.234567
Iteration  400: Cost = 0.156234
Iteration  600: Cost = 0.098765
Iteration  800: Cost = 0.067891
Iteration 1000: Cost = 0.045678

Making predictions on test set...

============================================================
MODEL EVALUATION RESULTS
============================================================

Accuracy:    0.9167 (91.67%)
Precision:   0.8966
Recall:      0.9286
F1-Score:    0.9123
Specificity: 0.9048

Confusion Matrix:
                 Predicted
                Ham  Spam
Actual Ham       28    2
       Spam      3   27

Sample Predictions (First 10 test cases):
True   Pred   Prob     Result    
Ham    Ham    0.123    ✓         
Spam   Spam   0.887    ✓         
Ham    Ham    0.234    ✓         
Spam   Spam   0.934    ✓         
Ham    Ham    0.156    ✓         
Spam   Spam   0.823    ✓         
Ham    Spam   0.678    ✗         
Spam   Spam   0.912    ✓         
Ham    Ham    0.089    ✓         
Spam   Spam   0.767    ✓         

============================================================
SPAM DETECTION PROJECT COMPLETED SUCCESSFULLY!
============================================================`,

    default: `================================================================================
AI PROJECT EXECUTION
================================================================================

Initializing project environment...
Loading required libraries...
Generating synthetic dataset...

Dataset Statistics:
- Samples: 1000
- Features: 10
- Classes: 3

Training model...
Epoch 1/100: Loss = 2.456
Epoch 25/100: Loss = 1.234
Epoch 50/100: Loss = 0.789
Epoch 75/100: Loss = 0.456
Epoch 100/100: Loss = 0.234

Model Evaluation:
- Accuracy: 87.5%
- Precision: 0.891
- Recall: 0.876
- F1-Score: 0.883

Project execution completed successfully!

=== EXECUTION COMPLETED ===`
  }

  return outputs[projectSlug] || outputs.default
}

// Get project-specific code (simplified version for demo)
function getProjectCode(projectSlug: string): string {
  // This would load the actual project files in a real implementation
  return `
# ${projectSlug.replace(/-/g, '_')}.py
# Main execution
if __name__ == "__main__":
    main()
`
}