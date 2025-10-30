'use client'

import React, { useState, useEffect } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface PCAVizProps {
  isRunning: boolean
  step: number
}

export function PCAViz({ isRunning, step }: PCAVizProps) {
  const [pcProgress, setPcProgress] = useState(0)
  
  const varianceData = [
    { pc: 'PC1', variance: 30.2, cumulative: 30.2 },
    { pc: 'PC2', variance: 19.1, cumulative: 49.3 },
    { pc: 'PC3', variance: 12.3, cumulative: 61.6 },
    { pc: 'PC4', variance: 8.6, cumulative: 70.2 },
    { pc: 'PC5', variance: 6.2, cumulative: 76.4 }
  ]

  const scatterData = [
    ...Array(50).fill(0).map(() => ({
      pc1: Math.random() * 6 - 3 + 2,
      pc2: Math.random() * 5 - 2.5 + 2,
      cluster: 0
    })),
    ...Array(50).fill(0).map(() => ({
      pc1: Math.random() * 4 - 2,
      pc2: Math.random() * 4 - 2,
      cluster: 1
    })),
    ...Array(50).fill(0).map(() => ({
      pc1: Math.random() * 5 - 2.5 - 2,
      pc2: Math.random() * 5 - 2.5 - 1.5,
      cluster: 2
    }))
  ]

  const COLORS = ['#22c55e', '#3b82f6', '#f59e0b']

  useEffect(() => {
    if (step >= 3 && isRunning) {
      const interval = setInterval(() => {
        setPcProgress(prev => Math.min(5, prev + 0.2))
      }, 200)
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing PCA"
      case 1: return "📈 Loading High-D Data"
      case 2: return "📊 Computing Covariance"
      case 3: return "🧮 Finding Principal Components"
      case 4: return "🔍 Eigenvalue Decomposition"
      case 5: return "⬇️ Reducing Dimensions"
      case 6: return "🗺️ Visualizing 2D Space"
      case 7: return "✅ PCA Complete!"
      default: return "PCA Analysis"
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-violet-50 to-purple-50">
        <CardHeader>
          <CardTitle className="text-xl">{getStepTitle()}</CardTitle>
          <Progress value={(step / 7) * 100} />
        </CardHeader>
      </Card>

      {step >= 1 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Dimensionality Reduction Challenge</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-violet-50 p-6 rounded-lg">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-red-100 p-4 rounded-lg border-2 border-red-300">
                  <div className="text-4xl font-bold text-red-700">150</div>
                  <div className="text-sm text-red-600">Original Dimensions</div>
                  <div className="text-xs text-gray-600 mt-1">⚠️ Hard to visualize</div>
                </div>
                <div className="flex items-center justify-center text-4xl text-purple-400">
                  →
                </div>
                <div className="bg-green-100 p-4 rounded-lg border-2 border-green-300">
                  <div className="text-4xl font-bold text-green-700">2</div>
                  <div className="text-sm text-green-600">Target Dimensions</div>
                  <div className="text-xs text-gray-600 mt-1">✅ Easy to visualize</div>
                </div>
              </div>
              <div className="mt-4 text-center text-sm text-gray-600">
                Reduction ratio: 98.7% | Retaining maximum variance
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 3 && (
        <Card>
          <CardHeader>
            <CardTitle>🧮 Computing Principal Components</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {varianceData.map((item, idx) => (
                <div key={item.pc} className={`p-3 rounded-lg border ${
                  idx < Math.floor(pcProgress) ? 'bg-purple-50 border-purple-300' : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold">{item.pc}</span>
                    <span className="text-sm">
                      Variance: {item.variance}% | Cumulative: {item.cumulative}%
                    </span>
                  </div>
                  <Progress value={idx < Math.floor(pcProgress) ? 100 : 0} />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Explained Variance</CardTitle>
            <CardDescription>How much information does each PC capture?</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={varianceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="pc" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="variance" fill="#8b5cf6" name="Variance (%)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 text-center p-3 bg-purple-50 rounded">
              <div className="text-sm text-purple-700">
                🎯 First 2 PCs capture <strong>49.3%</strong> of total variance
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>🗺️ 2D Projection: Customer Clusters</CardTitle>
            <CardDescription>150D data visualized in 2D space</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart>
                <CartesianGrid />
                <XAxis type="number" dataKey="pc1" name="PC1" label={{ value: 'Principal Component 1', position: 'insideBottom', offset: -5 }} />
                <YAxis type="number" dataKey="pc2" name="PC2" label={{ value: 'Principal Component 2', angle: -90, position: 'insideLeft' }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Cluster 1" data={scatterData.filter(d => d.cluster === 0)} fill={COLORS[0]} />
                <Scatter name="Cluster 2" data={scatterData.filter(d => d.cluster === 1)} fill={COLORS[1]} />
                <Scatter name="Cluster 3" data={scatterData.filter(d => d.cluster === 2)} fill={COLORS[2]} />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: COLORS[0] }} />
                <span className="text-sm">High-Value Shoppers</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: COLORS[1] }} />
                <span className="text-sm">Occasional Buyers</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: COLORS[2] }} />
                <span className="text-sm">Window Shoppers</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 7 && (
        <Card className="bg-gradient-to-r from-violet-50 to-purple-50 border-purple-200">
          <CardHeader>
            <CardTitle className="text-xl text-purple-800">🎉 Dimensionality Reduced!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">📊</div>
                <div className="font-semibold">150D → 2D</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🎯</div>
                <div className="font-semibold">49% Variance</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🔍</div>
                <div className="font-semibold">3 Clusters</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">⚡</div>
                <div className="font-semibold">Visualizable</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
