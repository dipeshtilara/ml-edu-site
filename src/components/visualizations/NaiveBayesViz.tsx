'use client'

import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface NaiveBayesVizProps {
  isRunning: boolean
  step: number
}

export function NaiveBayesViz({ isRunning, step }: NaiveBayesVizProps) {
  const [accuracy, setAccuracy] = useState(0)
  
  const classDistribution = [
    { name: 'Positive', value: 4234, color: '#22c55e' },
    { name: 'Negative', value: 3567, color: '#ef4444' },
    { name: 'Neutral', value: 2199, color: '#6b7280' }
  ]

  const topWords = [
    { sentiment: 'Positive', words: ['excellent', 'amazing', 'love', 'perfect', 'fantastic'] },
    { sentiment: 'Negative', words: ['terrible', 'awful', 'disappointing', 'waste', 'worst'] },
    { sentiment: 'Neutral', words: ['okay', 'average', 'normal', 'fine', 'acceptable'] }
  ]

  const performanceMetrics = [
    { class: 'Positive', precision: 91, recall: 93, f1: 92 },
    { class: 'Negative', precision: 89, recall: 88, f1: 88 },
    { class: 'Neutral', precision: 84, recall: 82, f1: 83 }
  ]

  useEffect(() => {
    if (step >= 5 && isRunning) {
      const interval = setInterval(() => {
        setAccuracy(prev => Math.min(89.3, prev + 2))
      }, 150)
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Sentiment Analysis"
      case 1: return "📝 Loading Product Reviews"
      case 2: return "🔤 Text Preprocessing"
      case 3: return "📊 Building Vocabulary"
      case 4: return "🧮 Training Naive Bayes"
      case 5: return "🎯 Calculating Probabilities"
      case 6: return "📈 Evaluating Model"
      case 7: return "✅ Sentiment Classifier Ready!"
      default: return "Sentiment Analysis"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-cyan-50">
        <CardHeader>
          <CardTitle className="text-xl flex items-center">
            {getStepTitle()}
          </CardTitle>
          <Progress value={(step / 7) * 100} className="w-full" />
        </CardHeader>
      </Card>

      {/* Step 1: Class Distribution */}
      {step >= 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">📊 Review Sentiment Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={classDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    dataKey="value"
                  >
                    {classDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              
              <div className="space-y-3">
                {classDistribution.map((item) => (
                  <div key={item.name} className="bg-gray-50 p-3 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold" style={{ color: item.color }}>{item.name}</span>
                      <span className="text-2xl font-bold">{item.value}</span>
                    </div>
                    <Progress value={(item.value / 10000) * 100} className="h-2" style={{ backgroundColor: item.color + '40' }} />
                    <div className="text-xs text-gray-600 mt-1">
                      {((item.value / 10000) * 100).toFixed(1)}% of reviews
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Top Words */}
      {step >= 3 && (
        <Card>
          <CardHeader>
            <CardTitle>🔤 Most Indicative Words per Sentiment</CardTitle>
            <CardDescription>
              Words with highest probability for each class
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {topWords.map((item) => (
                <div key={item.sentiment} className={`p-4 rounded-lg border-2 ${
                  item.sentiment === 'Positive' ? 'bg-green-50 border-green-300' :
                  item.sentiment === 'Negative' ? 'bg-red-50 border-red-300' :
                  'bg-gray-50 border-gray-300'
                }`}>
                  <div className={`text-lg font-bold mb-3 ${
                    item.sentiment === 'Positive' ? 'text-green-700' :
                    item.sentiment === 'Negative' ? 'text-red-700' :
                    'text-gray-700'
                  }`}>
                    {item.sentiment === 'Positive' ? '😊' : item.sentiment === 'Negative' ? '😞' : '😐'} {item.sentiment}
                  </div>
                  <div className="space-y-2">
                    {item.words.map((word, idx) => (
                      <div key={word} className="flex items-center">
                        <span className="text-sm font-mono bg-white px-2 py-1 rounded border">{word}</span>
                        <div className="ml-2 flex-1 h-2 bg-white rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              item.sentiment === 'Positive' ? 'bg-green-500' :
                              item.sentiment === 'Negative' ? 'bg-red-500' :
                              'bg-gray-500'
                            }`}
                            style={{ width: `${95 - idx * 8}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 5: Training Progress */}
      {step >= 5 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🧮 Training Progress
              {isRunning && accuracy < 89 && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-blue-50 p-6 rounded-lg">
              <div className="text-center mb-4">
                <div className="text-5xl font-bold text-blue-700">{accuracy.toFixed(1)}%</div>
                <div className="text-sm text-blue-600 mt-1">Model Accuracy</div>
              </div>
              <Progress value={accuracy} className="h-4" />
              <div className="mt-4 text-center text-sm text-gray-600">
                Vocabulary: 5,847 words | Training samples: 8,000
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 6: Performance Metrics */}
      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>📈 Per-Class Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={performanceMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="class" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="precision" fill="#3b82f6" name="Precision (%)" />
                <Bar dataKey="recall" fill="#10b981" name="Recall (%)" />
                <Bar dataKey="f1" fill="#8b5cf6" name="F1-Score (%)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Final Summary */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-blue-50 to-cyan-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-xl text-blue-800">💬 Sentiment Analysis Ready!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">📊</div>
                <div className="font-semibold text-blue-800">89.3% Accurate</div>
                <div className="text-sm text-blue-600">Test performance</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">🔤</div>
                <div className="font-semibold text-green-800">5,847 Words</div>
                <div className="text-sm text-green-600">Vocabulary size</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">⚡</div>
                <div className="font-semibold text-purple-800">Fast Training</div>
                <div className="text-sm text-purple-600">2.3 seconds</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-orange-200">
                <div className="text-3xl mb-2">🎯</div>
                <div className="font-semibold text-orange-800">Probabilistic</div>
                <div className="text-sm text-orange-600">Bayes theorem</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
