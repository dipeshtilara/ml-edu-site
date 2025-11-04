'use client'

import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface KNNVizProps {
  isRunning: boolean
  step: number
}

export function KNNViz({ isRunning, step }: KNNVizProps) {
  const [kValue, setKValue] = useState(1)
  const [accuracy, setAccuracy] = useState(0)

  const recommendations = [
    { user: 'User A', similarity: 95 },
    { user: 'User B', similarity: 88 },
    { user: 'User C', similarity: 82 },
    { user: 'User D', similarity: 76 },
    { user: 'User E', similarity: 71 }
  ]

  useEffect(() => {
    if (step >= 4 && isRunning) {
      const interval = setInterval(() => {
        setKValue(prev => Math.min(5, prev + 1))
        setAccuracy(prev => Math.min(87, prev + 3))
      }, 500)
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing K-NN"
      case 1: return "📚 Loading User-Item Data"
      case 2: return "📊 Computing Similarities"
      case 3: return "🔍 Finding Neighbors"
      case 4: return "🎯 Making Recommendations"
      case 5: return "📈 Evaluating Performance"
      case 6: return "✅ Model Ready!"
      case 7: return "🎉 Recommender System Active!"
      default: return "K-NN Recommender"
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50">
        <CardHeader>
          <CardTitle className="text-xl">{getStepTitle()}</CardTitle>
          <Progress value={(step / 7) * 100} />
        </CardHeader>
      </Card>

      {step >= 3 && (
        <Card>
          <CardHeader>
            <CardTitle>🔍 K Nearest Neighbors (K={kValue})</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={recommendations.slice(0, kValue)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="user" type="category" width={80} />
                <Tooltip />
                <Bar dataKey="similarity" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {step >= 5 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Recommendation Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-blue-50 p-6 rounded-lg">
              <div className="text-center mb-4">
                <div className="text-5xl font-bold text-blue-700">{accuracy}%</div>
                <div className="text-sm">Prediction Accuracy</div>
              </div>
              <Progress value={accuracy} className="h-4" />
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 7 && (
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50">
          <CardHeader>
            <CardTitle>🎉 K-NN Recommender Ready!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🎯</div>
                <div className="font-semibold">K=5 Optimal</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">📊</div>
                <div className="font-semibold">87% Accurate</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">⚡</div>
                <div className="font-semibold">Fast Lookup</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">👥</div>
                <div className="font-semibold">User-Based</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
