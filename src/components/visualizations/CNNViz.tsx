'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface CNNVizProps {
  isRunning: boolean
  step: number
}

export function CNNViz({ isRunning, step }: CNNVizProps) {
  const [epoch, setEpoch] = useState(0)
  const [accuracy, setAccuracy] = useState(32)

  const emotionPerformance = [
    { emotion: 'Happy', precision: 92, recall: 89 },
    { emotion: 'Sad', precision: 81, recall: 78 },
    { emotion: 'Angry', precision: 84, recall: 81 },
    { emotion: 'Surprise', precision: 87, recall: 90 },
    { emotion: 'Fear', precision: 79, recall: 77 },
    { emotion: 'Disgust', precision: 72, recall: 65 },
    { emotion: 'Neutral', precision: 83, recall: 87 }
  ]

  useEffect(() => {
    if (step >= 4 && isRunning && epoch < 50) {
      const interval = setInterval(() => {
        setEpoch(prev => Math.min(50, prev + 1))
        setAccuracy(prev => Math.min(84, 32 + (prev - 32) * 1.05 + 1.2))
      }, 150)
      return () => clearInterval(interval)
    }
  }, [step, isRunning, epoch])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing CNN"
      case 1: return "🖼️ Loading Facial Images"
      case 2: return "🔧 Building CNN Architecture"
      case 3: return "⚙️ Configuring Conv Layers"
      case 4: return "📊 Training CNN"
      case 5: return "🦾 Feature Extraction"
      case 6: return "🎯 Testing Emotions"
      case 7: return "✅ CNN Deployed!"
      default: return "CNN Training"
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-pink-50 to-rose-50">
        <CardHeader>
          <CardTitle className="text-xl">{getStepTitle()}</CardTitle>
          <Progress value={(step / 7) * 100} />
        </CardHeader>
      </Card>

      {step >= 2 && (
        <Card>
          <CardHeader>
            <CardTitle>🏗️ CNN Architecture Layers</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="bg-purple-100 p-3 rounded border-2 border-purple-300 text-center flex-1">
                  <div className="font-bold">Conv1</div>
                  <div className="text-xs">32 filters (3x3)</div>
                </div>
                <div className="text-lg">→</div>
                <div className="bg-blue-100 p-3 rounded border-2 border-blue-300 text-center flex-1">
                  <div className="font-bold">MaxPool</div>
                  <div className="text-xs">2x2</div>
                </div>
                <div className="text-lg">→</div>
                <div className="bg-cyan-100 p-3 rounded border-2 border-cyan-300 text-center flex-1">
                  <div className="font-bold">Conv2</div>
                  <div className="text-xs">64 filters (3x3)</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="bg-teal-100 p-3 rounded border-2 border-teal-300 text-center flex-1">
                  <div className="font-bold">MaxPool</div>
                  <div className="text-xs">2x2</div>
                </div>
                <div className="text-lg">→</div>
                <div className="bg-green-100 p-3 rounded border-2 border-green-300 text-center flex-1">
                  <div className="font-bold">Conv3</div>
                  <div className="text-xs">128 filters (3x3)</div>
                </div>
                <div className="text-lg">→</div>
                <div className="bg-emerald-100 p-3 rounded border-2 border-emerald-300 text-center flex-1">
                  <div className="font-bold">Dense</div>
                  <div className="text-xs">7 emotions</div>
                </div>
              </div>
            </div>
            <div className="mt-4 text-center text-sm text-gray-600">
              48x48 input → Feature maps → 7 emotion classes
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle>📋 Training Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-pink-50 p-4 rounded-lg border border-pink-200">
                <div className="text-3xl font-bold text-pink-700">{epoch}/50</div>
                <div className="text-sm">Epoch</div>
                <Progress value={(epoch / 50) * 100} className="mt-2" />
              </div>
              <div className="bg-rose-50 p-4 rounded-lg border border-rose-200">
                <div className="text-3xl font-bold text-rose-700">{accuracy.toFixed(1)}%</div>
                <div className="text-sm">Accuracy</div>
                <Progress value={accuracy} className="mt-2" />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>😊 Emotion Recognition Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={emotionPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="emotion" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="precision" fill="#ec4899" name="Precision (%)" />
                <Bar dataKey="recall" fill="#f43f5e" name="Recall (%)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {step >= 7 && (
        <Card className="bg-gradient-to-r from-pink-50 to-rose-50 border-pink-200">
          <CardHeader>
            <CardTitle className="text-xl text-pink-800">🎉 CNN Emotion Classifier Ready!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🖼️</div>
                <div className="font-semibold">84% Accurate</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🔍</div>
                <div className="font-semibold">Conv Layers</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">😊</div>
                <div className="font-semibold">7 Emotions</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">⚡</div>
                <div className="font-semibold">1.2M Params</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
