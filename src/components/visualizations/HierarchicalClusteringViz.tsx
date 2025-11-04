'use client'

import React, { useState, useEffect } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface HierarchicalClusteringVizProps {
  isRunning: boolean
  step: number
}

export function HierarchicalClusteringViz({ isRunning, step }: HierarchicalClusteringVizProps) {
  const [mergeStep, setMergeStep] = useState(0)
  const [clusters, setClusters] = useState(10)
  
  const geneData = Array.from({ length: 50 }, (_, i) => ({
    x: Math.random() * 10,
    y: Math.random() * 10,
    cluster: i % 3
  }))

  const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']

  useEffect(() => {
    if (step >= 4 && isRunning && mergeStep < 9) {
      const interval = setInterval(() => {
        setMergeStep(prev => prev + 1)
        setClusters(prev => Math.max(3, prev - 1))
      }, 800)
      return () => clearInterval(interval)
    }
  }, [step, isRunning, mergeStep])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Hierarchical Clustering"
      case 1: return "🧬 Loading Gene Expression Data"
      case 2: return "📊 Computing Distance Matrix"
      case 3: return "🔗 Building Linkage Matrix"
      case 4: return "🌳 Creating Dendrogram"
      case 5: return "📈 Agglomerative Clustering"
      case 6: return "🎯 Optimal Clusters Found"
      case 7: return "✅ Clustering Complete!"
      default: return "Hierarchical Clustering"
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-emerald-50 to-teal-50">
        <CardHeader>
          <CardTitle className="text-xl">{getStepTitle()}</CardTitle>
          <Progress value={(step / 7) * 100} />
        </CardHeader>
      </Card>

      {step >= 1 && (
        <Card>
          <CardHeader>
            <CardTitle>🧬 Gene Expression Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-emerald-50 p-4 rounded-lg">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-white p-3 rounded border">
                  <div className="text-2xl font-bold text-emerald-700">100</div>
                  <div className="text-xs">Genes</div>
                </div>
                <div className="bg-white p-3 rounded border">
                  <div className="text-2xl font-bold text-blue-700">20</div>
                  <div className="text-xs">Samples</div>
                </div>
                <div className="bg-white p-3 rounded border">
                  <div className="text-2xl font-bold text-purple-700">2000</div>
                  <div className="text-xs">Data Points</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle>🌳 Dendrogram Construction</CardTitle>
            <CardDescription>Hierarchical tree showing gene relationships</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-gradient-to-b from-emerald-50 to-white p-6 rounded-lg">
              <div className="text-center mb-4">
                <div className="text-3xl font-bold text-emerald-700">{clusters}</div>
                <div className="text-sm text-gray-600">Current Clusters</div>
              </div>
              <div className="space-y-2">
                {Array.from({ length: Math.min(mergeStep + 1, 5) }).map((_, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="flex-1 bg-emerald-200 h-2 rounded" />
                    <div className="text-xs text-gray-600">Merge {i + 1}</div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Gene Clusters</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid />
                <XAxis dataKey="x" name="Expression Level 1" />
                <YAxis dataKey="y" name="Expression Level 2" />
                <Tooltip />
                {[0, 1, 2].map((cluster) => (
                  <Scatter
                    key={cluster}
                    data={geneData.filter(d => d.cluster === cluster)}
                    fill={COLORS[cluster]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {step >= 7 && (
        <Card className="bg-gradient-to-r from-emerald-50 to-teal-50">
          <CardHeader>
            <CardTitle>🎉 Hierarchical Clustering Complete!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🌳</div>
                <div className="font-semibold">Dendrogram</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🧬</div>
                <div className="font-semibold">3 Clusters</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">📊</div>
                <div className="font-semibold">100 Genes</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🔗</div>
                <div className="font-semibold">Ward Linkage</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
