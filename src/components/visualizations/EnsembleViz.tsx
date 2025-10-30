'use client'

import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface EnsembleVizProps {
  isRunning: boolean
  step: number
}

export function EnsembleViz({ isRunning, step }: EnsembleVizProps) {
  const [modelsTrained, setModelsTrained] = useState(0)
  const [ensembleAccuracy, setEnsembleAccuracy] = useState(0)

  const individualModels = [
    { model: 'Tree 1', accuracy: 82.5, trained: modelsTrained >= 1 },
    { model: 'Tree 2', accuracy: 81.0, trained: modelsTrained >= 2 },
    { model: 'Tree 3', accuracy: 82.0, trained: modelsTrained >= 3 },
    { model: 'LogReg 1', accuracy: 84.5, trained: modelsTrained >= 4 },
    { model: 'LogReg 2', accuracy: 85.0, trained: modelsTrained >= 5 }
  ]

  useEffect(() => {
    if (step >= 4 && isRunning && modelsTrained < 5) {
      const interval = setInterval(() => {
        setModelsTrained(prev => Math.min(5, prev + 1))
      }, 800)
      return () => clearInterval(interval)
    }
  }, [step, isRunning, modelsTrained])

  useEffect(() => {
    if (step >= 5 && modelsTrained === 5) {
      const interval = setInterval(() => {
        setEnsembleAccuracy(prev => Math.min(88.5, prev + 1.5))
      }, 100)
      return () => clearInterval(interval)
    }
  }, [step, modelsTrained])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Ensemble"
      case 1: return "🏭 Loading Health Data"
      case 2: return "🔧 Creating Model Pool"
      case 3: return "⚙️ Configuring Voting"
      case 4: return "🧮 Training 5 Models"
      case 5: return "🤝 Combining Predictions"
      case 6: return "🎯 Evaluating Ensemble"
      case 7: return "✅ Ensemble Ready!"
      default: return "Ensemble Learning"
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-teal-50 to-cyan-50">
        <CardHeader>
          <CardTitle className="text-xl">{getStepTitle()}</CardTitle>
          <Progress value={(step / 7) * 100} />
        </CardHeader>
      </Card>

      {step >= 2 && (
        <Card>
          <CardHeader>
            <CardTitle>🔧 Ensemble Architecture</CardTitle>
            <CardDescription>Combining multiple models for better predictions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-teal-50 p-6 rounded-lg">
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="bg-green-100 p-3 rounded border-2 border-green-300 text-center flex-1">
                    <div className="font-bold text-sm">Decision Tree #1</div>
                  </div>
                  <div className="bg-green-100 p-3 rounded border-2 border-green-300 text-center flex-1">
                    <div className="font-bold text-sm">Decision Tree #2</div>
                  </div>
                  <div className="bg-green-100 p-3 rounded border-2 border-green-300 text-center flex-1">
                    <div className="font-bold text-sm">Decision Tree #3</div>
                  </div>
                </div>
                <div className="text-center text-2xl text-teal-400">↓</div>
                <div className="flex items-center gap-2">
                  <div className="bg-blue-100 p-3 rounded border-2 border-blue-300 text-center flex-1">
                    <div className="font-bold text-sm">Logistic Reg #1</div>
                  </div>
                  <div className="bg-blue-100 p-3 rounded border-2 border-blue-300 text-center flex-1">
                    <div className="font-bold text-sm">Logistic Reg #2</div>
                  </div>
                </div>
                <div className="text-center text-2xl text-teal-400">↓</div>
                <div className="bg-purple-100 p-4 rounded-lg border-2 border-purple-300 text-center">
                  <div className="font-bold">Voting Classifier</div>
                  <div className="text-xs text-purple-600 mt-1">Majority vote decides final prediction</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🧮 Training Individual Models
              {isRunning && modelsTrained < 5 && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {individualModels.map((model, idx) => (
                <div key={model.model} className={`p-4 rounded-lg border-2 ${
                  model.trained ? 'bg-green-50 border-green-300' : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">{model.model}</span>
                    {model.trained ? (
                      <span className="text-green-600 font-semibold">✓ Trained</span>
                    ) : (
                      <span className="text-gray-400">Pending...</span>
                    )}
                  </div>
                  {model.trained && (
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Accuracy:</span>
                        <span className="font-bold">{model.accuracy}%</span>
                      </div>
                      <Progress value={model.accuracy} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 5 && modelsTrained === 5 && (
        <Card>
          <CardHeader>
            <CardTitle>🤝 Ensemble Performance Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="text-sm font-semibold mb-3 text-center">Individual Models</div>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={individualModels}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" angle={-45} textAnchor="end" height={80} />
                    <YAxis domain={[75, 90]} />
                    <Tooltip />
                    <Bar dataKey="accuracy" fill="#6b7280" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div>
                <div className="text-sm font-semibold mb-3 text-center">Ensemble Power</div>
                <div className="bg-gradient-to-br from-green-50 to-emerald-100 p-6 rounded-lg border-2 border-green-300 h-[250px] flex flex-col items-center justify-center">
                  <div className="text-6xl font-bold text-green-700 mb-2">{ensembleAccuracy.toFixed(1)}%</div>
                  <div className="text-sm text-green-600 mb-4">Ensemble Accuracy</div>
                  <Progress value={ensembleAccuracy} className="w-full h-3" />
                  <div className="mt-4 text-xs text-center text-green-700">
                    🎉 <strong>+{(ensembleAccuracy - 83).toFixed(1)}%</strong> improvement over average
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>🎯 Confusion Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
              <div className="bg-green-100 p-6 rounded-lg border-2 border-green-500 text-center">
                <div className="text-4xl font-bold text-green-700">98</div>
                <div className="text-sm text-green-600 mt-1">✅ True Healthy</div>
              </div>
              <div className="bg-orange-100 p-6 rounded-lg border-2 border-orange-300 text-center">
                <div className="text-4xl font-bold text-orange-700">7</div>
                <div className="text-sm text-orange-600 mt-1">❌ False Alarm</div>
              </div>
              <div className="bg-red-100 p-6 rounded-lg border-2 border-red-300 text-center">
                <div className="text-4xl font-bold text-red-700">9</div>
                <div className="text-sm text-red-600 mt-1">❌ Missed Disease</div>
              </div>
              <div className="bg-green-100 p-6 rounded-lg border-2 border-green-500 text-center">
                <div className="text-4xl font-bold text-green-700">86</div>
                <div className="text-sm text-green-600 mt-1">✅ Caught Disease</div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 text-center">
              <div className="bg-blue-50 p-3 rounded">
                <div className="text-xl font-bold text-blue-700">90.5%</div>
                <div className="text-xs text-blue-600">Sensitivity</div>
              </div>
              <div className="bg-purple-50 p-3 rounded">
                <div className="text-xl font-bold text-purple-700">93.3%</div>
                <div className="text-xs text-purple-600">Specificity</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 7 && (
        <Card className="bg-gradient-to-r from-teal-50 to-cyan-50 border-teal-200">
          <CardHeader>
            <CardTitle className="text-xl text-teal-800">🤝 Ensemble Classifier Ready!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🤝</div>
                <div className="font-semibold">5 Models</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">📈</div>
                <div className="font-semibold">88.5% Accurate</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">⬆️</div>
                <div className="font-semibold">+5.5% Gain</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🗳️</div>
                <div className="font-semibold">Hard Voting</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
