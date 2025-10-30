'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Area, AreaChart } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface RandomForestVizProps {
  isRunning: boolean
  step: number
}

export function RandomForestViz({ isRunning, step }: RandomForestVizProps) {
  const [treeProgress, setTreeProgress] = useState(0)
  const [oobScore, setOobScore] = useState(0)
  const [featureImportance, setFeatureImportance] = useState<any[]>([])
  const [predictions, setPredictions] = useState<any[]>([])
  const [accuracy, setAccuracy] = useState(0)

  // Generate feature importance
  useEffect(() => {
    if (step >= 3) {
      const features = [
        { feature: '30-Day MA', importance: 18.4, icon: '📊' },
        { feature: 'RSI', importance: 15.7, icon: '📈' },
        { feature: 'MACD', importance: 13.2, icon: '📉' },
        { feature: '7-Day MA', importance: 11.9, icon: '📊' },
        { feature: 'Momentum', importance: 10.5, icon: '⚡' },
        { feature: 'Bollinger', importance: 9.8, icon: '📏' }
      ]
      setFeatureImportance(features)
    }
  }, [step])

  // Simulate tree training
  useEffect(() => {
    if (step >= 4 && isRunning) {
      const interval = setInterval(() => {
        setTreeProgress(prev => Math.min(100, prev + 2))
        setOobScore(prev => Math.min(0.8945, prev + 0.02))
      }, 100)
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  // Generate predictions
  useEffect(() => {
    if (step >= 6) {
      const predData = Array.from({ length: 20 }, (_, i) => ({
        day: i + 1,
        actual: 145 + Math.random() * 15,
        predicted: 145 + Math.random() * 15
      }))
      setPredictions(predData)
      setAccuracy(89.45)
    }
  }, [step])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Stock Price Prediction"
      case 1: return "📊 Loading Historical Stock Data"
      case 2: return "🔍 Analyzing Technical Indicators"
      case 3: return "🌳 Feature Importance Analysis"
      case 4: return "🌲 Training Random Forest (100 Trees)"
      case 5: return "📈 Ensemble Learning in Progress"
      case 6: return "🎯 Making Stock Predictions"
      case 7: return "✅ Trading Strategy Ready"
      default: return "Random Forest Analysis"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-emerald-50 to-teal-50">
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
            <CardTitle className="text-lg">📈 Stock Market Prediction Challenge</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-emerald-50 p-6 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-4">📊 ➜ 🌳🌳🌳 ➜ 💰</div>
                <p className="text-lg font-semibold text-emerald-800">
                  Historical Data → 100 Decision Trees → Price Prediction
                </p>
                <p className="text-emerald-600 mt-2">
                  Random Forest combines multiple trees for more accurate stock predictions!
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Feature Importance */}
      {step >= 3 && featureImportance.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>🎯 Which Indicators Matter Most?</CardTitle>
            <CardDescription>
              Feature importance for stock price prediction
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureImportance} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={100} />
                <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
                <Bar dataKey="importance" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 text-sm text-gray-600 bg-blue-50 p-3 rounded">
              💡 <strong>30-Day Moving Average</strong> is the strongest predictor,
              followed by RSI and MACD technical indicators.
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Tree Training */}
      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🌲 Building Forest of 100 Trees
              {isRunning && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
            <CardDescription>
              Each tree learns from different data samples
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="text-center mb-4">
                  <div className="text-5xl font-bold text-emerald-600">
                    {treeProgress}/100
                  </div>
                  <div className="text-sm text-gray-600">Trees Trained</div>
                </div>
                <Progress value={treeProgress} className="h-4" />
                
                <div className="mt-6 bg-emerald-50 p-4 rounded-lg">
                  <div className="text-lg font-semibold text-emerald-800 mb-2">
                    Out-of-Bag Score: {(oobScore * 100).toFixed(2)}%
                  </div>
                  <Progress value={oobScore * 100} className="h-2" />
                  <p className="text-xs text-emerald-600 mt-2">
                    Measures accuracy on data not used for training each tree
                  </p>
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <div className="text-2xl font-bold text-blue-700">100</div>
                  <div className="text-sm text-blue-600">Decision Trees</div>
                </div>
                
                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                  <div className="text-2xl font-bold text-purple-700">87,543</div>
                  <div className="text-sm text-purple-600">Total Tree Nodes</div>
                </div>
                
                <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                  <div className="text-2xl font-bold text-orange-700">12.3</div>
                  <div className="text-sm text-orange-600">Average Tree Depth</div>
                </div>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-yellow-50 rounded border border-yellow-200">
              <div className="text-xs text-yellow-800">
                🌳 <strong>Why 100 Trees?</strong> More trees = Better predictions but slower training.
                100 is the sweet spot for accuracy vs. speed!
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 6: Predictions */}
      {step >= 6 && predictions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>📈 Actual vs Predicted Prices</CardTitle>
              <CardDescription>
                How accurate are our stock predictions?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={predictions}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" label={{ value: 'Trading Day', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Stock Price ($)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Price']} />
                  <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} name="Actual Price" />
                  <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" name="Predicted" />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                🎯 Blue = Actual prices | Green = Random Forest predictions
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>📊 Performance Metrics</CardTitle>
              <CardDescription>
                Model accuracy and trading results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-green-800">R² Score</span>
                    <span className="text-2xl font-bold text-green-700">{accuracy.toFixed(1)}%</span>
                  </div>
                  <Progress value={accuracy} className="mt-2" />
                </div>
                
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-blue-800">Direction Accuracy</span>
                    <span className="text-2xl font-bold text-blue-700">86.0%</span>
                  </div>
                  <Progress value={86} className="mt-2" />
                </div>
                
                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-purple-800">Mean Abs Error</span>
                    <span className="text-2xl font-bold text-purple-700">$2.34</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-3 bg-emerald-50 rounded border border-emerald-200">
                    <div className="text-xl font-bold text-emerald-700">+32.4%</div>
                    <div className="text-xs text-emerald-600">Portfolio Return</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 rounded border border-orange-200">
                    <div className="text-xl font-bold text-orange-700">1.87</div>
                    <div className="text-xs text-orange-600">Sharpe Ratio</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Final Insights */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <CardHeader>
            <CardTitle className="text-xl text-green-800">🎉 Random Forest Trading System Deployed!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">🌳</div>
                <div className="font-semibold text-green-800">Ensemble Power</div>
                <div className="text-sm text-green-600">100 trees voting together</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">📊</div>
                <div className="font-semibold text-blue-800">89% Accurate</div>
                <div className="text-sm text-blue-600">R² prediction score</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">⚡</div>
                <div className="font-semibold text-purple-800">Fast Predictions</div>
                <div className="text-sm text-purple-600">Real-time trading signals</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-orange-200">
                <div className="text-3xl mb-2">💰</div>
                <div className="font-semibold text-orange-800">+32% Returns</div>
                <div className="text-sm text-orange-600">Simulated profit</div>
              </div>
            </div>
            
            <div className="p-4 bg-white rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">🎓 Why Random Forest Works for Stocks:</h4>
              <ul className="space-y-2 text-sm text-green-700">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Each tree learns from different data samples (bootstrap sampling)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>100 trees vote to make final prediction - reduces individual errors</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Handles non-linear relationships between price and indicators</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Feature importance helps identify which indicators drive predictions</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
