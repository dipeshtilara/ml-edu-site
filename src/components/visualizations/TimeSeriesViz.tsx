'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface TimeSeriesVizProps {
  isRunning: boolean
  step: number
}

export function TimeSeriesViz({ isRunning, step }: TimeSeriesVizProps) {
  const [forecastDays, setForecastDays] = useState(0)
  
  const weeklyPattern = [
    { day: 'Mon', multiplier: 0.82 },
    { day: 'Tue', multiplier: 0.87 },
    { day: 'Wed', multiplier: 0.93 },
    { day: 'Thu', multiplier: 0.98 },
    { day: 'Fri', multiplier: 1.05 },
    { day: 'Sat', multiplier: 1.32 },
    { day: 'Sun', multiplier: 1.43 }
  ]

  const historicalData = Array.from({ length: 30 }, (_, i) => ({
    day: i + 1,
    sales: 100 + i * 0.5 + Math.sin(i / 7 * Math.PI * 2) * 20 + Math.random() * 10
  }))

  const forecastData = Array.from({ length: 14 }, (_, i) => ({
    day: 31 + i,
    actual: 115 + i * 0.5 + Math.sin((30 + i) / 7 * Math.PI * 2) * 20,
    forecast: 115 + i * 0.5 + Math.sin((30 + i) / 7 * Math.PI * 2) * 20 + (Math.random() - 0.5) * 8
  }))

  useEffect(() => {
    if (step >= 5 && isRunning) {
      const interval = setInterval(() => {
        setForecastDays(prev => Math.min(14, prev + 1))
      }, 300)
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Forecasting"
      case 1: return "📈 Loading Time Series"
      case 2: return "🔍 Decomposition Analysis"
      case 3: return "📊 Detecting Seasonality"
      case 4: return "🧮 Training Model"
      case 5: return "🔮 Generating Forecasts"
      case 6: return "🎯 Evaluating Accuracy"
      case 7: return "✅ Forecasting Ready!"
      default: return "Time Series Analysis"
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-to-r from-amber-50 to-orange-50">
        <CardHeader>
          <CardTitle className="text-xl">{getStepTitle()}</CardTitle>
          <Progress value={(step / 7) * 100} />
        </CardHeader>
      </Card>

      {step >= 2 && (
        <Card>
          <CardHeader>
            <CardTitle>📈 Historical Sales Data (Last 30 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" label={{ value: 'Day', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Sales ($)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Sales']} />
                <Line type="monotone" dataKey="sales" stroke="#f59e0b" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-3 gap-3 text-center">
              <div className="bg-amber-50 p-3 rounded">
                <div className="text-2xl font-bold text-amber-700">$142.56</div>
                <div className="text-xs text-amber-600">Average Sales</div>
              </div>
              <div className="bg-orange-50 p-3 rounded">
                <div className="text-2xl font-bold text-orange-700">+$0.52</div>
                <div className="text-xs text-orange-600">Daily Growth</div>
              </div>
              <div className="bg-yellow-50 p-3 rounded">
                <div className="text-2xl font-bold text-yellow-700">7 days</div>
                <div className="text-xs text-yellow-600">Cycle Period</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 3 && (
        <Card>
          <CardHeader>
            <CardTitle>📊 Weekly Seasonality Pattern</CardTitle>
            <CardDescription>Sales multiplier by day of week</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={weeklyPattern}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis domain={[0.7, 1.5]} />
                <Tooltip formatter={(value) => [`${Number(value).toFixed(2)}x`, 'Multiplier']} />
                <Bar dataKey="multiplier" radius={[8, 8, 0, 0]}>
                  {weeklyPattern.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.multiplier > 1 ? '#22c55e' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 p-3 bg-green-50 rounded text-center text-sm">
              <div className="font-semibold text-green-700">📈 Peak Days: Saturday & Sunday (+32-43%)</div>
              <div className="text-green-600 text-xs mt-1">📉 Low Days: Monday & Tuesday (-18-13%)</div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 5 && (
        <Card>
          <CardHeader>
            <CardTitle>🔮 Sales Forecast (Next 14 Days)</CardTitle>
            <CardDescription>Actual vs Predicted values</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={forecastData.slice(0, forecastDays)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" label={{ value: 'Day', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Sales ($)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Sales']} />
                <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} name="Actual" />
                <Line type="monotone" dataKey="forecast" stroke="#22c55e" strokeWidth={2} strokeDasharray="5 5" name="Forecast" />
              </LineChart>
            </ResponsiveContainer>
            {forecastDays === 14 && (
              <div className="mt-4 text-center p-3 bg-blue-50 rounded">
                <div className="text-sm text-blue-700">
                  🎯 Forecast complete! MAPE: <strong>2.1%</strong> (highly accurate)
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {step >= 6 && (
        <Card>
          <CardHeader>
            <CardTitle>🎯 Forecast Accuracy Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="text-3xl font-bold text-green-700">2.1%</div>
                <div className="text-sm text-green-600">MAPE</div>
                <Progress value={97.9} className="mt-2" />
              </div>
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <div className="text-3xl font-bold text-blue-700">$3.45</div>
                <div className="text-sm text-blue-600">MAE</div>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                <div className="text-3xl font-bold text-purple-700">92.3%</div>
                <div className="text-sm text-purple-600">R² Score</div>
                <Progress value={92.3} className="mt-2" />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {step >= 7 && (
        <Card className="bg-gradient-to-r from-amber-50 to-orange-50 border-orange-200">
          <CardHeader>
            <CardTitle className="text-xl text-orange-800">📈 Forecasting System Ready!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">📈</div>
                <div className="font-semibold">Trend Detected</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🔁</div>
                <div className="font-semibold">Weekly Cycle</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🎯</div>
                <div className="font-semibold">2.1% Error</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg">
                <div className="text-3xl">🔮</div>
                <div className="font-semibold">36 Day Forecast</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
