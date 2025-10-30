'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface NeuralNetworkVizProps {
  isRunning: boolean
  step: number
}

export function NeuralNetworkViz({ isRunning, step }: NeuralNetworkVizProps) {
  const [epoch, setEpoch] = useState(0)
  const [trainingData, setTrainingData] = useState<any[]>([])
  const [accuracy, setAccuracy] = useState(0)
  const [loss, setLoss] = useState(2.0)

  // Simulate training progress
  useEffect(() => {
    if (step >= 4 && isRunning && epoch < 50) {
      const interval = setInterval(() => {
        setEpoch(prev => {
          const newEpoch = Math.min(50, prev + 1)
          const newLoss = 2.145 * Math.exp(-0.06 * newEpoch)
          const newAcc = 42.3 + (54.9 * (1 - Math.exp(-0.08 * newEpoch)))
          
          setLoss(newLoss)
          setAccuracy(newAcc)
          
          if (newEpoch % 5 === 0) {
            setTrainingData(prev => [...prev, {
              epoch: newEpoch,
              loss: newLoss,
              accuracy: newAcc,
              valAccuracy: newAcc + Math.random() * 2 - 1
            }])
          }
          
          return newEpoch
        })
      }, 150)
      return () => clearInterval(interval)
    }
  }, [step, isRunning, epoch])

  const perDigitAccuracy = [
    { digit: '0', accuracy: 98.5 },
    { digit: '1', accuracy: 99.1 },
    { digit: '2', accuracy: 96.8 },
    { digit: '3', accuracy: 96.2 },
    { digit: '4', accuracy: 97.5 },
    { digit: '5', accuracy: 96.4 },
    { digit: '6', accuracy: 97.9 },
    { digit: '7', accuracy: 96.8 },
    { digit: '8', accuracy: 95.9 },
    { digit: '9', accuracy: 96.1 }
  ]

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Deep Learning"
      case 1: return "🖼️ Loading MNIST Dataset"
      case 2: return "🔧 Building Neural Network"
      case 3: return "⚙️ Configuring Architecture"
      case 4: return "🧠 Training Neural Network"
      case 5: return "📊 Backpropagation in Progress"
      case 6: return "🎯 Evaluating Model Performance"
      case 7: return "✅ Neural Network Trained!"
      default: return "Neural Network Training"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-purple-50 to-indigo-50">
        <CardHeader>
          <CardTitle className="text-xl flex items-center">
            {getStepTitle()}
          </CardTitle>
          <Progress value={(step / 7) * 100} className="w-full" />
        </CardHeader>
      </Card>

      {/* Step 2: Architecture */}
      {step >= 2 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">🏗️ Neural Network Architecture</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-gradient-to-b from-purple-50 to-indigo-50 p-6 rounded-lg">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="bg-purple-100 p-4 rounded-lg border-2 border-purple-300 flex-1 mx-2">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-700">784</div>
                      <div className="text-sm text-purple-600">Input Layer</div>
                      <div className="text-xs text-gray-600">28×28 pixels</div>
                    </div>
                  </div>
                  
                  <div className="text-2xl text-purple-400">→</div>
                  
                  <div className="bg-blue-100 p-4 rounded-lg border-2 border-blue-300 flex-1 mx-2">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-700">128</div>
                      <div className="text-sm text-blue-600">Hidden Layer 1</div>
                      <div className="text-xs text-gray-600">ReLU</div>
                    </div>
                  </div>
                  
                  <div className="text-2xl text-blue-400">→</div>
                  
                  <div className="bg-cyan-100 p-4 rounded-lg border-2 border-cyan-300 flex-1 mx-2">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-cyan-700">64</div>
                      <div className="text-sm text-cyan-600">Hidden Layer 2</div>
                      <div className="text-xs text-gray-600">ReLU</div>
                    </div>
                  </div>
                  
                  <div className="text-2xl text-cyan-400">→</div>
                  
                  <div className="bg-green-100 p-4 rounded-lg border-2 border-green-300 flex-1 mx-2">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-700">10</div>
                      <div className="text-sm text-green-600">Output Layer</div>
                      <div className="text-xs text-gray-600">Softmax</div>
                    </div>
                  </div>
                </div>
                
                <div className="text-center text-sm text-gray-600 mt-4">
                  <div className="font-semibold">Total Parameters: 109,386</div>
                  <div className="text-xs mt-1">Forward propagation → Backward propagation → Weight updates</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Training Progress */}
      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              📈 Training Progress
              {isRunning && epoch < 50 && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
            <CardDescription>
              Epoch {epoch}/50 - Backpropagation with Gradient Descent
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
                <div className="text-3xl font-bold text-blue-700">{epoch}/50</div>
                <div className="text-sm text-blue-600">Current Epoch</div>
                <Progress value={(epoch / 50) * 100} className="mt-2" />
              </div>
              
              <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg border border-red-200">
                <div className="text-3xl font-bold text-red-700">{loss.toFixed(3)}</div>
                <div className="text-sm text-red-600">Loss (decreasing)</div>
                <Progress value={100 - (loss / 2.145) * 100} className="mt-2" />
              </div>
              
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                <div className="text-3xl font-bold text-green-700">{accuracy.toFixed(1)}%</div>
                <div className="text-sm text-green-600">Accuracy</div>
                <Progress value={accuracy} className="mt-2" />
              </div>
            </div>
            
            {trainingData.length > 0 && (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis yAxisId="left" label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                  <YAxis yAxisId="right" orientation="right" label={{ value: 'Loss', angle: 90, position: 'insideRight' }} />
                  <Tooltip />
                  <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} name="Train Accuracy" />
                  <Line yAxisId="left" type="monotone" dataKey="valAccuracy" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" name="Val Accuracy" />
                  <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 6: Per-Digit Performance */}
      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>🔢 Per-Digit Classification Accuracy</CardTitle>
            <CardDescription>
              How well does the network recognize each digit?
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={perDigitAccuracy}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="digit" />
                <YAxis domain={[90, 100]} />
                <Tooltip formatter={(value) => [`${value}%`, 'Accuracy']} />
                <Bar dataKey="accuracy" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            
            <div className="mt-4 grid grid-cols-5 gap-2">
              {perDigitAccuracy.map((item) => (
                <div key={item.digit} className="text-center p-2 bg-purple-50 rounded">
                  <div className="text-2xl font-bold text-purple-700">{item.digit}</div>
                  <div className="text-xs text-purple-600">{item.accuracy}%</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Final Summary */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-purple-50 to-indigo-50 border-purple-200">
          <CardHeader>
            <CardTitle className="text-xl text-purple-800">🎉 Neural Network Training Complete!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">🧠</div>
                <div className="font-semibold text-purple-800">Deep Learning</div>
                <div className="text-sm text-purple-600">Multi-layer network</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">📊</div>
                <div className="font-semibold text-blue-800">97.2% Accuracy</div>
                <div className="text-sm text-blue-600">Test performance</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">⚡</div>
                <div className="font-semibold text-green-800">Backprop</div>
                <div className="text-sm text-green-600">Gradient descent</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-orange-200">
                <div className="text-3xl mb-2">🎯</div>
                <div className="font-semibold text-orange-800">109K Params</div>
                <div className="text-sm text-orange-600">Trained weights</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
