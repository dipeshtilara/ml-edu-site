'use client'

import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface SVMVizProps {
  isRunning: boolean
  step: number
}

export function SVMViz({ isRunning, step }: SVMVizProps) {
  const [classifiers, setClassifiers] = useState<any[]>([])
  const [confusionMatrix, setConfusionMatrix] = useState<any>(null)
  const [accuracy, setAccuracy] = useState(0)

  // Simulate classifier training
  useEffect(() => {
    if (step >= 4 && isRunning) {
      const digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      let currentDigit = 0
      
      const interval = setInterval(() => {
        if (currentDigit < 10) {
          setClassifiers(prev => [...prev, {
            digit: currentDigit,
            supportVectors: 90 + Math.floor(Math.random() * 80),
            accuracy: 96 + Math.random() * 3.5,
            iterations: 200 + Math.floor(Math.random() * 150)
          }])
          currentDigit++
        }
      }, 800)
      
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  // Generate confusion matrix
  useEffect(() => {
    if (step >= 6) {
      const matrix = [
        [19, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 19, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 19, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 19, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 17, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 20, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 19, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 18, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 20]
      ]
      setConfusionMatrix(matrix)
      setAccuracy(97.5)
    }
  }, [step])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Digit Recognition"
      case 1: return "✍️ Loading Handwritten Digits"
      case 2: return "🔍 Analyzing Pixel Features"
      case 3: return "⚙️ Preparing SVM Classifiers"
      case 4: return "🧠 Training 10 Binary Classifiers"
      case 5: return "📊 Optimizing Decision Boundaries"
      case 6: return "🎯 Testing Recognition Accuracy"
      case 7: return "✅ Digit Recognition System Ready"
      default: return "SVM Digit Recognition"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-indigo-50 to-purple-50">
        <CardHeader>
          <CardTitle className="text-xl flex items-center">
            {getStepTitle()}
          </CardTitle>
          <Progress value={(step / 7) * 100} className="w-full" />
        </CardHeader>
      </Card>

      {/* Step 1: Problem Understanding */}
      {step >= 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">✍️ Handwriting Recognition Challenge</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-indigo-50 p-6 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-4">📝 ➜ 🤖 ➜ 🔢</div>
                <p className="text-lg font-semibold text-indigo-800">
                  Handwritten Image → SVM Classifier → Recognized Digit
                </p>
                <p className="text-indigo-600 mt-2">
                  Support Vector Machines find the optimal boundary between digit classes!
                </p>
              </div>
              
              <div className="mt-6 bg-white p-4 rounded-lg border-2 border-indigo-200">
                <div className="text-center">
                  <div className="text-sm text-gray-600 mb-2">Sample Digit Visualization:</div>
                  <div className="font-mono text-xs leading-tight bg-gray-100 p-3 rounded inline-block">
                    ╔══════════╗<br/>
                    ║ ░░████░░ ║<br/>
                    ║ ██░░░░██ ║<br/>
                    ║ ░░░░░░██ ║<br/>
                    ║ ░░░░██░░ ║<br/>
                    ║ ░░██░░░░ ║<br/>
                    ║ ██░░░░██ ║<br/>
                    ║ ░░████░░ ║<br/>
                    ╚══════════╝
                  </div>
                  <div className="text-sm text-indigo-600 mt-2">
                    8x8 pixels = 64 features per digit
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Classifier Training */}
      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🧠 Training Binary SVM Classifiers
              {isRunning && classifiers.length < 10 && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
            <CardDescription>
              One-vs-Rest strategy: Each classifier specializes in one digit
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {classifiers.map((classifier, idx) => (
                <div key={idx} className="bg-white p-4 rounded-lg border-2 border-indigo-100">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className="text-3xl font-bold text-indigo-600 bg-indigo-100 w-12 h-12 rounded-lg flex items-center justify-center">
                        {classifier.digit}
                      </div>
                      <div>
                        <div className="font-semibold text-gray-800">
                          Digit {classifier.digit} vs Rest
                        </div>
                        <div className="text-sm text-gray-600">
                          Support Vectors: {classifier.supportVectors} | 
                          Accuracy: {classifier.accuracy.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="text-green-600 font-semibold">
                      ✓ Converged
                    </div>
                  </div>
                  <Progress value={classifier.accuracy} className="h-2" />
                </div>
              ))}
              
              {classifiers.length < 10 && isRunning && (
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 animate-pulse">
                  <div className="text-center text-blue-700">
                    Training classifier {classifiers.length + 1}/10...
                  </div>
                </div>
              )}
            </div>
            
            {classifiers.length === 10 && (
              <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-green-700">10</div>
                    <div className="text-sm text-green-600">Classifiers</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-700">
                      {Math.round(classifiers.reduce((sum, c) => sum + c.supportVectors, 0))}
                    </div>
                    <div className="text-sm text-green-600">Support Vectors</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-700">
                      {(classifiers.reduce((sum, c) => sum + c.accuracy, 0) / 10).toFixed(1)}%
                    </div>
                    <div className="text-sm text-green-600">Avg Accuracy</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 6: Confusion Matrix */}
      {step >= 6 && confusionMatrix && (
        <Card>
          <CardHeader>
            <CardTitle>🎯 Confusion Matrix</CardTitle>
            <CardDescription>
              Classification accuracy for each digit (rows = actual, columns = predicted)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr>
                    <th className="border p-2 bg-gray-100"></th>
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(d => (
                      <th key={d} className="border p-2 bg-gray-100 font-bold">{d}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.map((row: number[], actualDigit: number) => (
                    <tr key={actualDigit}>
                      <td className="border p-2 bg-gray-100 font-bold text-center">{actualDigit}</td>
                      {row.map((count, predictedDigit) => (
                        <td 
                          key={predictedDigit} 
                          className={`border p-2 text-center ${
                            actualDigit === predictedDigit 
                              ? 'bg-green-100 font-bold text-green-700' 
                              : count > 0 
                                ? 'bg-red-100 text-red-700'
                                : 'bg-white text-gray-400'
                          }`}
                        >
                          {count || '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="text-2xl font-bold text-green-700">{accuracy}%</div>
                <div className="text-sm text-green-600">Overall Accuracy</div>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <div className="text-2xl font-bold text-blue-700">195/200</div>
                <div className="text-sm text-blue-600">Correct Predictions</div>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <div className="text-2xl font-bold text-orange-700">5</div>
                <div className="text-sm text-orange-600">Misclassifications</div>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-yellow-50 rounded border border-yellow-200">
              <div className="text-sm text-yellow-800">
                <strong>Common Confusions:</strong> Digit 3 ↔ 5 (similar curves), 
                4 ↔ 9 (vertical strokes), 7 ↔ 9 (diagonals)
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Per-Digit Performance */}
      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Per-Digit Performance</CardTitle>
            <CardDescription>
              Precision and recall for each digit class
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { digit: '0', precision: 95, recall: 100 },
                { digit: '1', precision: 100, recall: 100 },
                { digit: '2', precision: 100, recall: 95 },
                { digit: '3', precision: 95, recall: 95 },
                { digit: '4', precision: 100, recall: 95 },
                { digit: '5', precision: 94, recall: 94 },
                { digit: '6', precision: 100, recall: 100 },
                { digit: '7', precision: 100, recall: 95 },
                { digit: '8', precision: 95, recall: 100 },
                { digit: '9', precision: 95, recall: 100 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="digit" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="precision" fill="#3b82f6" name="Precision %" />
                <Bar dataKey="recall" fill="#10b981" name="Recall %" />
              </BarChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4 mt-4 text-sm">
              <div className="flex items-center">
                <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
                <span>Precision (How often we're right)</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
                <span>Recall (How many we catch)</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Final Insights */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200">
          <CardHeader>
            <CardTitle className="text-xl text-indigo-800">✍️ SVM Digit Recognition Deployed!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-4 bg-white rounded-lg border border-indigo-200">
                <div className="text-3xl mb-2">🎯</div>
                <div className="font-semibold text-indigo-800">97.5% Accurate</div>
                <div className="text-sm text-indigo-600">High recognition rate</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">⚡</div>
                <div className="font-semibold text-blue-800">Fast Recognition</div>
                <div className="text-sm text-blue-600">Instant predictions</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">🧠</div>
                <div className="font-semibold text-purple-800">RBF Kernel</div>
                <div className="text-sm text-purple-600">Non-linear classification</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">📊</div>
                <div className="font-semibold text-green-800">16.8% SVs</div>
                <div className="text-sm text-green-600">Memory efficient</div>
              </div>
            </div>
            
            <div className="p-4 bg-white rounded-lg border border-indigo-200">
              <h4 className="font-semibold text-indigo-800 mb-2">🎓 Why SVM Excels at Digit Recognition:</h4>
              <ul className="space-y-2 text-sm text-indigo-700">
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">✓</span>
                  <span>Finds optimal decision boundary with maximum margin between classes</span>
                </li>
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">✓</span>
                  <span>RBF kernel handles non-linear patterns in handwriting</span>
                </li>
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">✓</span>
                  <span>Support vectors capture only the most important examples</span>
                </li>
                <li className="flex items-start">
                  <span className="text-indigo-500 mr-2">✓</span>
                  <span>One-vs-Rest strategy scales well to 10-class problem</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
