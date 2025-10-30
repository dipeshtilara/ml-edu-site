'use client'

import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface DecisionTreeVizProps {
  isRunning: boolean
  step: number
}

export function DecisionTreeViz({ isRunning, step }: DecisionTreeVizProps) {
  const [treeMetrics, setTreeMetrics] = useState({ accuracy: 0, depth: 0, nodes: 0 })
  const [featureImportance, setFeatureImportance] = useState<any[]>([])
  const [confusionData, setConfusionData] = useState<any>(null)

  const COLORS = ['#22c55e', '#ef4444', '#3b82f6', '#f59e0b']

  // Generate feature importance data
  useEffect(() => {
    if (step >= 4) {
      const features = [
        { feature: 'Blood Pressure', importance: 85, icon: '💉' },
        { feature: 'Age', importance: 72, icon: '👤' },
        { feature: 'Cholesterol', importance: 68, icon: '🩺' },
        { feature: 'Heart Rate', importance: 61, icon: '❤️' },
        { feature: 'BMI', importance: 54, icon: '⚖️' },
        { feature: 'Exercise Hours', importance: 48, icon: '🏃' }
      ]
      setFeatureImportance(features)
    }
  }, [step])

  // Simulate tree growth
  useEffect(() => {
    if (step >= 5 && isRunning) {
      const interval = setInterval(() => {
        setTreeMetrics(prev => ({
          accuracy: Math.min(92, prev.accuracy + 1.5),
          depth: Math.min(8, prev.depth + 0.2),
          nodes: Math.min(127, prev.nodes + 3)
        }))
      }, 200)
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  // Generate confusion matrix
  useEffect(() => {
    if (step >= 6) {
      setConfusionData({
        tp: 42,
        tn: 38,
        fp: 4,
        fn: 6,
        classes: ['Healthy', 'At Risk', 'Disease', 'Critical']
      })
    }
  }, [step])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Medical Diagnosis System"
      case 1: return "🏥 Understanding Medical Decision Making"
      case 2: return "📊 Loading Patient Data"
      case 3: return "🌳 Building Decision Tree Structure"
      case 4: return "🧠 Calculating Feature Importance"
      case 5: return "📈 Training Decision Tree Model"
      case 6: return "🎯 Evaluating Diagnosis Accuracy"
      case 7: return "✅ Medical AI System Ready"
      default: return "Decision Tree Analysis"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-green-50 to-emerald-50">
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
            <CardTitle className="text-lg">🏥 How Decision Trees Work in Medicine</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-green-50 p-6 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-4">🩺 ➜ 🌳 ➜ 💊</div>
                <p className="text-lg font-semibold text-green-800">
                  Patient Symptoms → Decision Tree → Diagnosis & Treatment
                </p>
                <p className="text-green-600 mt-2">
                  Like a doctor's thought process, the tree asks questions to reach a diagnosis!
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Tree Structure */}
      {step >= 3 && (
        <Card>
          <CardHeader>
            <CardTitle>🌳 Decision Tree Structure</CardTitle>
            <CardDescription>
              Visualizing how the AI makes medical decisions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-50 p-6 rounded-lg">
              <div className="text-center mb-6">
                <div className="inline-block bg-blue-100 text-blue-800 px-6 py-3 rounded-lg mb-4 font-semibold">
                  🩺 Root: Blood Pressure {'>'} 140?
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center">
                  <div className="bg-orange-100 text-orange-800 px-4 py-2 rounded-lg mb-2 text-sm font-medium">
                    ✓ Yes: Age {'>'} 60?
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-red-50 text-red-800 p-2 rounded border border-red-200">
                      ✓ → Critical
                    </div>
                    <div className="bg-yellow-50 text-yellow-800 p-2 rounded border border-yellow-200">
                      ✗ → At Risk
                    </div>
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="bg-green-100 text-green-800 px-4 py-2 rounded-lg mb-2 text-sm font-medium">
                    ✗ No: Cholesterol {'>'} 200?
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-yellow-50 text-yellow-800 p-2 rounded border border-yellow-200">
                      ✓ → Monitor
                    </div>
                    <div className="bg-green-50 text-green-800 p-2 rounded border border-green-200">
                      ✗ → Healthy
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="text-xs text-gray-600 text-center mt-4">
                💡 Each node asks a yes/no question until we reach a final diagnosis (leaf node)
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Feature Importance */}
      {step >= 4 && featureImportance.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Which Medical Factors Matter Most?</CardTitle>
            <CardDescription>
              Feature importance for accurate diagnosis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureImportance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={120} />
                <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
                <Bar dataKey="importance" fill="#22c55e" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 text-sm text-gray-600 bg-blue-50 p-3 rounded">
              💡 <strong>Blood Pressure</strong> is the most critical factor for diagnosis, 
              followed by age and cholesterol levels.
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 5: Tree Metrics */}
      {step >= 5 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              📈 Model Training Progress
              {isRunning && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
                <div className="text-3xl font-bold text-green-700">
                  {treeMetrics.accuracy.toFixed(1)}%
                </div>
                <div className="text-sm text-green-600">Diagnosis Accuracy</div>
                <Progress value={treeMetrics.accuracy} className="mt-2" />
              </div>
              
              <div className="bg-gradient-to-br from-blue-50 to-sky-50 p-4 rounded-lg border border-blue-200">
                <div className="text-3xl font-bold text-blue-700">
                  {Math.round(treeMetrics.depth)}
                </div>
                <div className="text-sm text-blue-600">Tree Depth (levels)</div>
                <Progress value={(treeMetrics.depth / 8) * 100} className="mt-2" />
              </div>
              
              <div className="bg-gradient-to-br from-purple-50 to-indigo-50 p-4 rounded-lg border border-purple-200">
                <div className="text-3xl font-bold text-purple-700">
                  {Math.round(treeMetrics.nodes)}
                </div>
                <div className="text-sm text-purple-600">Total Decision Nodes</div>
                <Progress value={(treeMetrics.nodes / 127) * 100} className="mt-2" />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 6: Confusion Matrix */}
      {step >= 6 && confusionData && (
        <Card>
          <CardHeader>
            <CardTitle>🎯 Diagnosis Performance Matrix</CardTitle>
            <CardDescription>
              How accurate are our medical predictions?
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-green-100 p-6 rounded-lg border-2 border-green-500 text-center">
                    <div className="text-4xl font-bold text-green-700">{confusionData.tn}</div>
                    <div className="text-sm text-green-600 mt-1">✅ Correct Healthy</div>
                  </div>
                  <div className="bg-red-100 p-6 rounded-lg border-2 border-red-300 text-center">
                    <div className="text-4xl font-bold text-red-700">{confusionData.fp}</div>
                    <div className="text-sm text-red-600 mt-1">❌ False Alarm</div>
                  </div>
                  <div className="bg-orange-100 p-6 rounded-lg border-2 border-orange-300 text-center">
                    <div className="text-4xl font-bold text-orange-700">{confusionData.fn}</div>
                    <div className="text-sm text-orange-600 mt-1">❌ Missed Disease</div>
                  </div>
                  <div className="bg-green-100 p-6 rounded-lg border-2 border-green-500 text-center">
                    <div className="text-4xl font-bold text-green-700">{confusionData.tp}</div>
                    <div className="text-sm text-green-600 mt-1">✅ Caught Disease</div>
                  </div>
                </div>
                
                <div className="text-sm space-y-1 bg-gray-50 p-4 rounded">
                  <div className="flex justify-between">
                    <span>Overall Accuracy:</span>
                    <span className="font-semibold text-green-600">
                      {(((confusionData.tp + confusionData.tn) / (confusionData.tp + confusionData.tn + confusionData.fp + confusionData.fn)) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Sensitivity (Recall):</span>
                    <span className="font-semibold text-blue-600">
                      {((confusionData.tp / (confusionData.tp + confusionData.fn)) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Specificity:</span>
                    <span className="font-semibold text-purple-600">
                      {((confusionData.tn / (confusionData.tn + confusionData.fp)) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Correct Predictions', value: confusionData.tp + confusionData.tn, color: '#22c55e' },
                        { name: 'Errors', value: confusionData.fp + confusionData.fn, color: '#ef4444' }
                      ]}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      dataKey="value"
                    >
                      {[
                        { name: 'Correct', color: '#22c55e' },
                        { name: 'Errors', color: '#ef4444' }
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="text-sm text-gray-600 text-center mt-4">
                  🩺 Decision Tree successfully diagnoses 89% of cases correctly!
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Final Insights */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <CardHeader>
            <CardTitle className="text-xl text-green-800">🏥 Medical AI System Deployed!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">🌳</div>
                <div className="font-semibold text-green-800">Interpretable</div>
                <div className="text-sm text-green-600">Easy to understand decisions</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">⚡</div>
                <div className="font-semibold text-blue-800">Fast</div>
                <div className="text-sm text-blue-600">Quick diagnosis</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">🎯</div>
                <div className="font-semibold text-purple-800">Accurate</div>
                <div className="text-sm text-purple-600">89%+ accuracy</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-orange-200">
                <div className="text-3xl mb-2">🩺</div>
                <div className="font-semibold text-orange-800">Medical Grade</div>
                <div className="text-sm text-orange-600">Healthcare ready</div>
              </div>
            </div>
            
            <div className="p-4 bg-white rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">🎓 What Decision Trees Teach Us:</h4>
              <ul className="space-y-2 text-sm text-green-700">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Decision trees mimic human decision-making with if-then rules</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>They're highly interpretable - we can see exactly why the AI made a choice</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Feature importance helps doctors understand which factors matter most</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Perfect for medical diagnosis where transparency is critical</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
